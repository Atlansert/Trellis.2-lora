"""Synchronous checkpoint groups committed only by a complete marker.

No migration of legacy files. No overwrite/cleanup: interrupted files are kept
as evidence; save into a new run directory after a partial write. Files are
atomic individually, not a multi-file transaction: only the final marker makes
a group visible. Rank0 RNG is not a DDP/prefetched-worker exact-resume promise.

LoRA 微调实验的检查点契约（同步、以“完成标记”为准）：
- 一组检查点 = 若干 <name>_stepNNNNNNN.pt 文件 + 一个 complete_stepNNNNNNN.json 标记；
  只有标记写成功，该 step 才算提交完成。
- 单个文件写临时文件后 os.replace 保证原子；整组不是事务，中途失败会保留残留文件作为证据，
  因此不允许就地重写同一 step，须换新的 run 目录。
- 每个文件记录 size + sha256 以便校验；validate_checkpoint 会核对 schema/step/文件名/大小/哈希。
- RNG 状态只保存 rank0，不承诺 DDP 或 dataloader worker 的精确复现。
"""
import copy
import hashlib
import json
import os
import random
import re
import uuid
import warnings
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'trellis.checkpoint-group.v1'


def marker_name(step):
    """完成标记文件名（7 位零填充），非法 step 直接报错。"""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('step must be nonnegative integer')
    return f'complete_step{step:07d}.json'


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(directory):
    if os.name == 'posix':
        fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def cpu_snapshot(value):
    """Owned CPU copies; never alias live optimizer/EMA/model CPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu').clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _check_names(names, step):
    suffix = f'_step{step:07d}.pt'
    if not names or len(names) != len(set(names)):
        raise ValueError('nonempty unique group filenames required')
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name or not name.endswith(suffix):
            raise ValueError(f'unsafe/wrong-step filename: {name}')
    if f'misc_step{step:07d}.pt' not in names:
        raise ValueError('checkpoint group must include misc')
    if not any(not n.startswith('misc_') for n in names):
        raise ValueError('checkpoint group must include model weights')


def atomic_save_group(ckpt_dir, step, payloads, metadata=None, non_blocking=True):
    """Compatibility argument non_blocking is accepted; always completes sync.

    A cooperative exclusive reservation prevents concurrent writers for this
    step. Reservation and partial files are deliberately retained on failure.
    """
    directory = Path(ckpt_dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / marker_name(step)
    names = list(payloads)
    _check_names(names, step)
    targets = [directory / n for n in names] + [marker]
    if any(p.exists() for p in targets):
        raise FileExistsError('checkpoint group targets already exist; preserve old/partial files and use a new run directory')
    lock = directory / f'.checkpoint_step{step:07d}.reservation'
    # 预约文件用 'x' 独占创建，避免同一 step 被并发写；失败时故意保留
    with open(lock, 'x', encoding='utf-8') as f:
        f.write('synchronous writer; do not retry this step in-place after failure\n')
        f.flush()
        os.fsync(f.fileno())
    # CPU clones prevent CPU state_dict aliases even if callers pass live data.
    owned = cpu_snapshot(payloads)
    files = []
    token = uuid.uuid4().hex
    for name, payload in owned.items():
        path = directory / name
        tmp = directory / f'.{name}.{token}.tmp'
        with open(tmp, 'xb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise FileExistsError(f'checkpoint target appeared while saving: {path}')
        os.replace(tmp, path)  # 原子替换：写完临时文件再改名
        files.append({'name': name, 'size_bytes': path.stat().st_size, 'sha256': _hash(path)})
    _fsync_dir(directory)
    manifest = {'schema': SCHEMA, 'step': step, 'files': files,
                'metadata': metadata or {}, 'synchronous': True,
                'requested_non_blocking': bool(non_blocking)}
    tmp = directory / f'.{marker.name}.{token}.tmp'
    with open(tmp, 'x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if marker.exists():
        raise FileExistsError('complete marker appeared while saving')
    os.replace(tmp, marker)
    _fsync_dir(directory)
    return manifest

# Stable alias for callers creating test fixtures.
save_checkpoint_group = atomic_save_group


def validate_checkpoint(ckpt_dir, step, verify_hashes=True):
    """校验某 step 的完成标记与文件：schema/step、文件名、大小与（可选）sha256。"""
    directory = Path(ckpt_dir)
    marker = directory / marker_name(step)
    if not marker.is_file():
        raise ValueError(f'checkpoint complete marker missing: {marker}')
    try:
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA or type(manifest.get('step')) is not int or manifest['step'] != step:
            raise ValueError('checkpoint marker schema/step mismatch')
        files = manifest['files']
        names = [rec['name'] for rec in files]
        _check_names(names, step)
        for rec in files:
            path = directory / rec['name']
            if type(rec['size_bytes']) is not int or rec['size_bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', rec['sha256']):
                raise ValueError('invalid checkpoint file record')
            if not path.is_file() or path.is_symlink() or path.stat().st_size != rec['size_bytes']:
                raise ValueError(f'checkpoint file missing/size mismatch: {path}')
            if verify_hashes and _hash(path) != rec['sha256']:
                raise ValueError(f'checkpoint file hash mismatch: {path}')
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as e:
        raise ValueError(f'invalid checkpoint marker {marker}: {e}') from e
    return manifest


validate_complete_checkpoint = validate_checkpoint


def list_complete_steps(ckpt_dir, verify_hashes=False):
    """列出目录下所有已提交完成的 step（逐个校验标记，返回升序去重结果）。"""
    directory = Path(ckpt_dir)
    steps = []
    if not directory.exists():
        return steps
    for marker in directory.glob('complete_step*.json'):
        match = re.fullmatch(r'complete_step(\d{7,})\.json', marker.name)
        if not match:
            raise ValueError(f'invalid complete marker filename: {marker}')
        step = int(match.group(1))
        validate_checkpoint(directory, step, verify_hashes=verify_hashes)
        steps.append(step)
    return sorted(set(steps))


# LoRA 场景：只把非 LoRA 主体转成目标 dtype，LoRA 适配器保持 FP32 且 Parameter 身份不变
def cast_model_preserving_lora_and_complex(model, dtype):
    """Cast non-LoRA base as before; retain exact FP32 adapters and complex buffers.

    Restore original unrounded storage, not rounded bf16.float(); Parameter
    identities must remain intact for AMP master lists and optimizer references.
    """
    from ..modules.lora import LoRALinear
    adapters = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for layer in (module.lora_down, module.lora_up):
                for name, param in layer.named_parameters(recurse=False):
                    adapters.append((layer, name, param, param.detach()))
    complex_buffers = []
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.is_complex():
                complex_buffers.append((module, name, buffer))
                module._buffers[name] = None
    try:
        model.to(dtype=dtype)
    finally:
        for module, name, buffer in complex_buffers:
            # Current module device is unchanged by a dtype-only cast.
            module._buffers[name] = buffer
        for layer, name, original_param, data in adapters:
            if getattr(layer, name) is not original_param:
                raise RuntimeError('dtype cast replaced LoRA Parameter identity; AMP/optimizer references invalid')
            original_param.data = data.to(device=original_param.device, dtype=torch.float32)


# 优化器状态里的浮点张量（动量等）对齐到参数自身的 device/dtype，step 保持标量
def align_optimizer_moments(optimizer):
    """Floating tensor moments match owning parameter; scalar step stays scalar."""
    for param, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != 'step' and value.is_floating_point():
                state[key] = value.to(device=param.device, dtype=param.dtype)


def capture_rng_state(rank=0, world_size=1):
    """抓取当前 RNG 状态（python/numpy/torch CPU 及可用的 CUDA），仅供 rank0。"""
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch_cpu': torch.get_rng_state().clone(), 'rank': rank,
             'world_size': world_size,
             'scope': 'rank0-only; sampler prefetch/worker state not captured; not exact trajectory guarantee'}
    if torch.cuda.is_available():
        state['torch_cuda'] = [x.clone().cpu() for x in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state, rank=0, world_size=1):
    """恢复 RNG；rank/world_size 或 CUDA 设备数不匹配时给警告并降级返回 False。"""
    if state is None:
        warnings.warn('Legacy checkpoint has no RNG state; resume is not exact', RuntimeWarning)
        return False
    if rank != state.get('rank'):
        warnings.warn('Rank0 RNG not restored on another rank; DDP RNG is not exact', RuntimeWarning)
        return False
    if world_size != state.get('world_size') or world_size > 1:
        warnings.warn('Rank0-only RNG checkpoint: DDP topology/other ranks/worker state not exact', RuntimeWarning)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if 'torch_cuda' in state:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            warnings.warn('CUDA RNG device topology differs; CUDA state not restored', RuntimeWarning)
        else:
            torch.cuda.set_rng_state_all(state['torch_cuda'])
    return True


def snapshot_allowed(dataset, final=False):
    """是否允许做快照；dataset.skip_snapshot=True 时跳过（latent 代理不等于几何验收）。"""
    if getattr(dataset, 'skip_snapshot', False):
        if final:
            print('Skipping final snapshot: dataset.skip_snapshot=True (latent proxy is not geometry acceptance).', flush=True)
        return False
    return True
