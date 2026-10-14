"""CPU-only training launch safety; checkpoint transactions owned by checkpoint_contract.

训练启动安全契约（只做 CPU 侧校验，检查点事务由 checkpoint_contract 负责）：
- resolve_checkpoint 在任何写输出之前决定本次从哪个 step 恢复：'none' / 'latest' / 数字 step，
  禁止把已有输出历史与新 run 混用；'latest' 没有合法完成标记时报错而不是拿 legacy misc 顶替。
- run_attempts 负责重试：auto_retry 表示最多尝试次数(0→1)；重试只能从本次输出目录里已提交的
  检查点恢复，否则宁可停下，也不重放不确定的失败轨迹。
- preserve_launch_records 把 config/command 与每次 launch 记录落盘（已存在的不覆盖）。
"""
import hashlib
import json
import os
import re
import traceback
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path


def _steps(directory):
    from .checkpoint_contract import list_complete_steps
    return list_complete_steps(directory, verify_hashes=False)


def _validate(directory, step):
    from .checkpoint_contract import validate_checkpoint
    return validate_checkpoint(directory, step, verify_hashes=True)


def expected_files(cfg, step):
    """某个 step 应存在的文件清单：misc + 每个模型本体及其各 ema_rate 副本。"""
    rates = cfg['trainer']['args'].get('ema_rate', 0.9999)
    if not isinstance(rates, (list, tuple)):
        rates = [rates]
    names = [f'misc_step{step:07d}.pt']
    for name in cfg['models']:
        names.append(f'{name}_step{step:07d}.pt')
        names.extend(f'{name}_ema{rate}_step{step:07d}.pt' for rate in rates)
    return names


def _has_history(root):
    root = Path(root)
    return (root / 'config.json').exists() or (root / 'command.txt').exists() or (
        (root / 'ckpts').is_dir() and any((root / 'ckpts').iterdir()))


def resolve_checkpoint(cfg, complete_steps=None, validate=None):
    """Resolve before any output writes. Never infer legacy commit from misc alone."""
    complete_steps = complete_steps or _steps
    validate = validate or _validate
    output = Path(cfg['output_dir']).resolve()
    load = Path(cfg.get('load_dir') or output).resolve()
    choice = str(cfg.get('ckpt', 'latest'))
    if load != output and _has_history(output):
        raise ValueError('external checkpoint resume requires a fresh output directory; existing output history cannot be mixed')
    cfg['load_dir'] = str(load)
    cfg['load_ckpt'] = None
    directory = load / 'ckpts'
    if choice == 'none':
        if _has_history(output):
            raise ValueError('fresh --ckpt none would overwrite existing output history; use a new directory or explicit resume')
        return cfg
    if choice == 'latest':
        # 有完整检查点就取最大 step；否则要求输出目录是全新的空目录
        steps = complete_steps(directory) if directory.is_dir() else []
        if steps:
            step = max(steps)
            validate(directory, step)
            _require_files(directory, expected_files(cfg, step))
            cfg['load_ckpt'] = step
            return cfg
        if _has_history(load):
            raise ValueError('latest has no valid atomic complete checkpoint; legacy misc is NOT a commit. Use an explicit numeric legacy step after integrity review, or a new output directory')
        if load != output or (load.exists() and any(load.iterdir())):
            raise ValueError('latest source is not a fresh empty output directory and has no complete checkpoint')
        print('[Resume] Fresh empty output; latest resolves to step0 (no checkpoint)', flush=True)
        return cfg
    if not re.fullmatch(r'\d+', choice) or int(choice) <= 0:
        raise ValueError('checkpoint must be latest, none, or a positive numeric step')
    step = int(choice)
    marker = directory / f'complete_step{step:07d}.json'
    # 显式数字 step：有完成标记就走完整校验，没有则按 legacy 只查文件名/大小并告警
    if marker.exists():
        validate(directory, step)
    else:
        warnings.warn('Explicit LEGACY checkpoint: no atomic marker; filename/size completeness checked but hashes, save transaction and RNG provenance cannot be certified', RuntimeWarning)
    _require_files(directory, expected_files(cfg, step))
    cfg['load_ckpt'] = step
    return cfg


def _require_files(directory, filenames):
    """检查文件齐全且非空；缺文件或 0 字节即视为该检查点组不完整。"""
    missing = [name for name in filenames if not (Path(directory)/name).is_file() or (Path(directory)/name).stat().st_size == 0]
    if missing:
        raise ValueError(f'incomplete checkpoint group: missing/empty {missing}')


def preserve_launch_records(cfg, config, command, source_path):
    """Called only after resolve; old config/command immutable, new attempts exclusive."""
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for name, text in [('config.json', json.dumps(config, indent=4)), ('command.txt', command + '\n')]:
        try:
            with (output/name).open('x', encoding='utf-8') as stream:
                stream.write(text)
        except FileExistsError:
            pass
    attempt_root = output/'run_attempts'
    attempt_root.mkdir(exist_ok=True)
    launch_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:12]
    record = {'schema':'trellis.train-launch.v1','launch_id':launch_id,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'command':command, 'source_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
              'requested_checkpoint':cfg['ckpt'],'resolved_checkpoint':cfg.get('load_ckpt'),
              'output_dir':str(output.resolve()),'load_dir':cfg['load_dir'],
              'config':config, 'attempts':[], 'status':'prepared',
              'max_attempts':max(1,int(cfg.get('auto_retry',3)))}
    path = attempt_root/(launch_id+'.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    return path, record


def _write_record(path, record):
    temp = Path(path).with_suffix('.json.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run_attempts(cfg, runner, record_path=None, record=None, complete_steps=None, validate=None):
    """auto_retry is historical MAX ATTEMPTS (0 => one), not extra retries.

    After failure only a checkpoint committed in THIS output can be resumed.
    No output commit => stop, even if initial external legacy resume was valid.
    This deliberately refuses replay of an uncertain failed optimizer trajectory.
    """
    record = record if record is not None else {'attempts':[]}
    count = max(1,int(cfg.get('auto_retry',3)))
    complete_steps = complete_steps or _steps
    validate = validate or _validate
    def persist():
        if record_path is not None:
            _write_record(record_path,record)
    for index in range(count):
        item = {'attempt':index+1,'load_dir':cfg['load_dir'],'checkpoint':cfg.get('load_ckpt'),
                'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
        record['attempts'].append(item);record['status']='running';persist()
        try:
            runner(cfg)
        except Exception as exc:
            item.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
                        finished_utc=datetime.now(timezone.utc).isoformat())
            if index+1 >= count:
                record.update(status='failed',finished_utc=item['finished_utc']);persist()
                raise RuntimeError(f'Training failed: exhausted {count} maximum attempts') from exc
            retry_cfg=dict(cfg,load_dir=cfg['output_dir'],ckpt='latest')
            try:
                directory=Path(cfg['output_dir'])/'ckpts'
                steps=complete_steps(directory) if directory.is_dir() else []
                if not steps:
                    raise ValueError('no complete checkpoint in current output; refusing restart from0 or replay of initial source')
                resolve_checkpoint(retry_cfg,complete_steps,validate)
            except Exception as resume_exc:
                record.update(status='failed',retry_refused=f'{type(resume_exc).__name__}: {resume_exc}',
                              finished_utc=datetime.now(timezone.utc).isoformat());persist()
                raise RuntimeError(f'Training failed; safe retry unavailable: {resume_exc}') from exc
            cfg.update(load_dir=retry_cfg['load_dir'],ckpt=retry_cfg['ckpt'],load_ckpt=retry_cfg['load_ckpt'])
            persist()
            print(f'[Retry] Next attempt {index+2}/{count}, committed current-output step {cfg["load_ckpt"]}',flush=True)
        else:
            item.update(status='success',finished_utc=datetime.now(timezone.utc).isoformat())
            record.update(status='success',finished_utc=item['finished_utc']);persist()
            return


def load_pretrained_checked(model,state):
    """Do not overwrite computed complex RoPE with historical serialized buffers.

    加载预训练权重：丢弃权重里的 rope_phases（由配置重新计算），
    除它之外的缺失 key 或任何多余 key 都直接报错。
    """
    state=dict(state)
    buffers=dict(model.named_buffers())
    if 'rope_phases' in buffers:
        state.pop('rope_phases',None)
    missing,unexpected=model.load_state_dict(state,strict=False)
    allowed={'rope_phases'} if 'rope_phases' in buffers else set()
    bad=set(missing)-allowed
    if bad or unexpected:
        raise RuntimeError(f'Pretrained mismatch: missing={sorted(bad)}, unexpected={unexpected}')
