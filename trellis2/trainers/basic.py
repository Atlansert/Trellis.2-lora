"""训练主循环 BasicTrainer：所有训练器的基类。

职责：初始化模型/DDP/主参数/EMA/优化器/调度器/弹性显存/梯度裁剪；检查点的"完整组"
原子保存与加载（模型+EMA+misc，带 complete marker 与哈希校验）；断点恢复（优化器、
数据采样进度、RNG 状态、scaler/log_scale）；训练步（梯度累积、混合精度、梯度裁剪、
NaN 保护）；EMA 更新、采样快照与日志/可视化。
输入：models、dataset 及一批训练超参；输出：output_dir 下的 ckpts/、samples/、tb_logs/、log.txt。
契约：save 只能由 rank0 调用；load / finetune_from / snapshot 需所有进程调用；
'inflat_all' 模式下更新发生在 FP32 主参数上；LoRA 与复数 buffer（如 rope_phases）
在 dtype 转换时必须保持原精度，rope_phases 不入库（可由模型自行重建）。
"""

from abc import abstractmethod
import os
import time
import json
import copy
import warnings
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils
from ..utils.checkpoint_contract import (atomic_save_group, validate_checkpoint, marker_name,
    cast_model_preserving_lora_and_complex, align_optimizer_moments, capture_rng_state,
    restore_rng_state, snapshot_allowed)


class BasicTrainer:
    """
    Trainer for basic training loop.
    
    训练主循环基类：负责初始化、检查点保存/恢复、训练步、EMA、快照与日志。
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        mix_precision_mode (str):
            - None: No mixed precision.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        mix_precision_dtype (str): Mixed precision dtype.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        parallel_mode (str): Parallel mode. Options are 'ddp'.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode=None,
        mix_precision_mode='inflat_all',
        mix_precision_dtype='float16',
        fp16_scale_growth=1e-3,
        parallel_mode='ddp',
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_dtype = 'float16'
            mix_precision_mode = fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.log = []
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck        

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        if self.parallel_mode == 'ddp' and self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - Mixed precision dtype: {self.mix_precision_dtype}')
        lines.append(f'  - Mixed precision mode: {self.mix_precision_mode}')
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            lines.append(f'  - FP16 scale growth: {self.fp16_scale_growth}')
        lines.append(f'  - Parallel mode: {self.parallel_mode}')
        return '\n'.join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        建 DDP 包装、主参数、EMA、优化器、LR 调度器、弹性显存控制器与梯度裁剪器。
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        # 需要梯度的参数汇总为 model_params；主参数按精度模式决定：
        # 'amp' 直接复用模型参数（由 GradScaler 处理），'inflat_all' 另建一份展平的 FP32 主参数
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        if self.mix_precision_mode == 'amp':
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'Mix precision mode {self.mix_precision_mode} is not implemented.')

        # Build EMA params
        # EMA 只在 rank0 上维护（落盘也只有 rank0），每个 ema_rate 对应一份深拷贝
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        # 先查 torch.optim，再回退到本模块 globals（支持自定义优化器）；优化器只更新主参数
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        # 同样先查 torch.optim.lr_scheduler，再回退 globals（例如 LinearWarmupLRScheduler）
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        # 弹性显存控制器：注册到声明为 Elastic 的模型上，训练中可按显存预算换入换出模块
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        # float/int 直接当阈值；dict 则实例化 grad_clip_utils 里的裁剪器（带状态，会被存进 misc）
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        可恢复采样器 ResumableSampler（保存/恢复数据遍历进度）+ 多进程 DataLoader。
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=min(16, max(1, int(np.ceil(os.cpu_count() / torch.cuda.device_count())))),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        把主参数写回各模型的 state_dict（inflat_all 下先拆回原形状），供保存使用。
        """
        if self.mix_precision_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        # RoPE phases are deterministic runtime buffers. Do not persist them:
        # 中文：RoPE 相位是可重建的运行时 buffer，不入库；否则 AMP 会把 complex64 相位转成 bf16 造成损坏。
        # AMP conversion can cast complex64 phases to bfloat16, corrupting
        # subsequent resume/inference.
        for state_dict in state_dicts.values():
            state_dict.pop('rope_phases', None)
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        从 state_dict 取回所有可训练参数并写回主参数（inflat_all 展平后拷贝，否则逐个 copy_）。
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.mix_precision_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        载入 load_dir/ckpts 下指定 step 的完整 checkpoint 组（所有进程都要调用）。
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        # 期望的完整组：每个模型一份权重 + 每个 EMA rate 一份 + misc（缺一不可）
        ckpt_dir = os.path.join(load_dir, 'ckpts')
        expected_names = {f'{name}_step{step:07d}.pt' for name in self.models}
        expected_names.update(f'{name}_ema{rate}_step{step:07d}.pt'
                              for name in self.models for rate in self.ema_rate)
        expected_names.add(f'misc_step{step:07d}.pt')
        if os.path.exists(os.path.join(ckpt_dir, marker_name(step))):
            manifest = validate_checkpoint(ckpt_dir, step, verify_hashes=True)
            if {item['name'] for item in manifest['files']} != expected_names:
                raise ValueError('checkpoint complete group does not match configured models/EMA rates')
        else:
            # Legacy is allowed only when explicitly selected by the caller;
            # 中文：无 complete marker 的历史 checkpoint 仅在调用方显式选择时允许，且要求文件齐全、非空
            # new latest selection uses complete markers, never misc alone.
            if not all(os.path.isfile(os.path.join(ckpt_dir, n)) and os.path.getsize(os.path.join(ckpt_dir, n)) > 0 for n in expected_names):
                raise ValueError('legacy checkpoint group incomplete')
            warnings.warn('Explicit legacy checkpoint: no atomic complete marker/hash contract', RuntimeWarning)
        model_ckpts = {}
        # 逐个模型载入权重；rope_phases 不入库，保留模型自建的 complex64 版本
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            # Keep the freshly initialized complex64 RoPE phases. Older
            # 中文：保留模型刚初始化的 complex64 RoPE 相位；老 checkpoint 里可能是被 AMP 转成 bf16 的副本
            # checkpoints may contain an AMP-cast bfloat16 copy.
            rope_phases = getattr(model, 'rope_phases', None)
            model_ckpt.pop('rope_phases', None)
            missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
            missing = [k for k in missing if k != 'rope_phases']
            if missing or unexpected:
                raise RuntimeError(f'Checkpoint {name} mismatch: missing={missing}, unexpected={unexpected}')
            if self.mix_precision_mode == 'amp':
                cast_model_preserving_lora_and_complex(model, self.mix_precision_dtype)
            if rope_phases is not None:
                model.rope_phases = rope_phases.to(device=self.device)
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        align_optimizer_moments(self.optimizer)
        if misc_ckpt['step'] != step:
            raise ValueError('misc checkpoint step does not match selected group')
        # 恢复 misc：步数已与所选组校验一致，下面依次恢复采样进度、精度状态、调度器、弹性控制器与 RNG
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        # RNG 状态按 rank 保存，保证断点续训的随机性与不间断训练一致
        restore_rng_state(misc_ckpt.get('rng_state'), rank=self.rank, world_size=self.world_size)
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self, non_blocking=True):
        """Commit a synchronous owned-CPU checkpoint group, rank0 only.

        non_blocking is retained for callers but no writer threads are used.
        Completion marker, not a misc file, is the transaction visibility point.
        中文：rank0 同步落盘一整组 checkpoint（模型 + EMA + misc），以 complete marker 作为提交点。
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving synchronous checkpoint at step {self.step}...', end='', flush=True)
        # 收集负载：模型参数 + 每个 EMA rate 的参数（都从主参数导出）
        payloads = {}
        for name, state in self._master_params_to_state_dicts(self.master_params).items():
            payloads[f'{name}_step{self.step:07d}.pt'] = state
        for i, rate in enumerate(self.ema_rate):
            for name, state in self._master_params_to_state_dicts(self.ema_params[i]).items():
                payloads[f'{name}_ema{rate}_step{self.step:07d}.pt'] = state
        # misc：续训所需的状态——优化器、步数、数据采样进度、RNG
        misc_ckpt = {'optimizer': self.optimizer.state_dict(), 'step': self.step,
                     'data_sampler': self.data_sampler.state_dict(),
                     'rng_state': capture_rng_state(rank=self.rank, world_size=self.world_size)}
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        # 再补齐精度状态/调度器/弹性控制器/裁剪器，然后整组原子写出
        payloads[f'misc_step{self.step:07d}.pt'] = misc_ckpt
        # 原子写整组 + 校验 + 最后写 marker；rng_scope 注明只覆盖 rank0（不含 dataloader worker 状态）
        atomic_save_group(os.path.join(self.output_dir, 'ckpts'), self.step, payloads,
                          metadata={'models': list(self.models), 'ema_rates': self.ema_rate,
                                    'rng_scope': 'rank0 only; prefetch/worker state not captured',
                                    'world_size': self.world_size}, non_blocking=non_blocking)
        # 记录已提交步数：run() 结束时据此判断是否需要补存最终 checkpoint
        self._last_committed_step = self.step
        print(' Done (complete marker committed).', flush=True)

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        从外部 checkpoint 初始化权重（键缺失或形状不符时跳过并告警，不报错）。
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=True)
                model_ckpt.pop('rope_phases', None)
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}
                model_ckpts[name] = model_ckpt
                missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
                missing = [k for k in missing if k != 'rope_phases']
                if missing or unexpected:
                    raise RuntimeError(f'Finetune checkpoint {name} mismatch: missing={missing}, unexpected={unexpected}')
                # Dtype-only conversion must preserve exact LoRA FP32 values and
                # 中文：这里只做 dtype 转换，必须保持 LoRA 的 FP32 数值与参数身份，并保留复数 buffer
                # parameter identity, plus every complex computed buffer.
                if self.mix_precision_mode == 'amp':
                    cast_model_preserving_lora_and_complex(model, self.mix_precision_dtype)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                # EMA 副本也用同一份初始权重（EMA 只在 rank0 维护）
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        抽象方法：由各训练器实现具体采样，返回待可视化的样本字典。
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        优先交给数据集自己的 visualize_sample；数据集没实现就原样返回。
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """
        Sample images from the dataset.
        直接从数据集采样若干样本拼图保存（用于确认数据加载/对齐是否正确）。
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[:min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            vis = self.visualize_sample(data)
            if isinstance(vis, dict):
                for k, v in vis.items():
                    if f'dataset_{k}' not in save_cfg:
                        save_cfg[f'dataset_{k}'] = []
                    save_cfg[f'dataset_{k}'].append(v)
            else:
                if 'dataset' not in save_cfg:
                    save_cfg['dataset'] = []
                save_cfg['dataset'].append(vis)
        for name, image in save_cfg.items():
            utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        快照流程：各进程分担样本 -> 可视化 -> 多卡 gather 到 rank0 -> 存图（rank0）。
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        with amp_context():
            samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master and not getattr(self.dataset, 'skip_snapshot', False):
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

        if self.is_master:
            print(' Done.')

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        EMA 更新：ema = ema*rate + master*(1-rate)，只在 rank0 上做。
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        分块 all_gather 主参数并逐一比对，确认各进程参数完全一致（防止静默不同步）。
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        抽象方法：子类实现具体损失，返回 (loss 字典, status 字典)。
        """
        pass

    def load_data(self):
        """
        Load data.
        取一个 batch 搬到设备；开启 prefetch 时提前预取下一批；再按 batch_split 切成子批（梯度累积用）。
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    def run_step(self, data_list):
        """
        Run a training step.
        一个训练步：各子批前向+反向累积梯度 -> 梯度裁剪 -> 优化器更新 -> 日志/EMA。
        """
        step_log = {'loss': {}, 'status': {}}
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        # 累积各子批（batch_split）的损失；非最后一个子批用 no_sync 抑制 DDP 提前同步
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                    self.scaler.scale(l).backward()
                elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        # 先把梯度换算回真实尺度（AMP 缩放 / inflat_all 的 2^log_scale），再统一裁剪
        if self.grad_clip is not None:
            if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
        ## step
        # 按精度模式更新参数；inflat_all 下要自己处理缩放与溢出（log_scale 自适应），
        # 梯度非有限就跳过本次更新（并降低 log_scale）
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.mix_precision_mode == 'inflat_all':
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2 ** self.log_scale
                if not any(not p.grad.isfinite().all() for p in self.model_params):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                else:
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any(not p.grad.isfinite().all() for p in self.master_params):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m')
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
        ## adjust learning rate
        # 先记录当前 lr 再让调度器步进
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for model_name, model in self.models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norms[f'{model_name}.{name}'] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[f'{model_name}.{name}'] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        # 每步更新 EMA（仅 rank0）
        if self.is_master:
            self.update_ema()

        return step_log

    def save_logs(self):
        # 追加写入 log.txt，并把本轮指标（跳过含 NaN 的项）汇总后写 TensorBoard
        log_str = '\n'.join([
            f'{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}' for step, log in self.log
        ])
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
            log_file.write(log_str + '\n')

        # show with mlflow
        log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
        log_show = dict_reduce(log_show, lambda x: np.mean(x))
        log_show = dict_flatten(log_show, sep='/')
        for key, value in log_show.items():
            self.writer.add_scalar(key, value, self.step)
        self.log = []
        
    def check_abort(self):
        """
        Check if training should be aborted due to certain conditions.
        中止条件：inflat_all 模式下 log_scale 降到 0 以下（说明数值发散），先存盘再抛错。
        """
        # 1. If log_scale in inflat_all mode is less than 0
        if self.mix_precision_dtype == torch.float16 and \
           self.mix_precision_mode == 'inflat_all' and \
           self.log_scale < 0:
            if self.is_master:
                print ('\n\n\033[91m')
                print (f'ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}.')
                print ('This indicates that the model is diverging. You should look into the model and the data.')
                print ('\033[0m')
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError('ABORT: log_scale in inflat_all mode is less than 0.')

    def run(self):
        """
        Run training.
        训练主循环：启动快照 -> 逐 step（取数据/训练/打印/DDP 校验/采样/日志/存盘）-> 收尾提交最终 checkpoint。
        """
        if self.is_master:
            print('\nStarting training...')
            try:
                self.snapshot_dataset(batch_size=self.snapshot_batch_size)
            except Exception as e:
                # Latent-only datasets (e.g. per-view SS LoRA) do not have
                # 中文：纯潜变量数据集（例如按视角的 SS LoRA 数据）没有图像形状的 x_0，无法拼图，跳过即可
                # image-shaped x_0 tensors suitable for torchvision grids.
                print(f'[Snapshot] skipped dataset visualization: {e}', flush=True)
        # 开始/恢复训练时先各出一张快照（数据集可声明 skip_snapshot 跳过）
        if not getattr(self.dataset, 'skip_snapshot', False):
            if self.step == 0:
                self.snapshot(suffix='init', batch_size=self.snapshot_batch_size)
            else: # resume
                self.snapshot(suffix=f'resume_step{self.step:07d}', batch_size=self.snapshot_batch_size)

        # 计时与打印节流（速度/ETA 按 i_print 区间统计）
        time_last_print = 0.0
        time_elapsed = 0.0
        # 主循环：断点恢复时 self.step 从已保存的步数继续
        while self.step < self.max_steps:
            time_start = time.time()

            # 取一个 batch（按 batch_split 切好）并完成一次参数更新
            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            # 周期性 DDP 一致性自检
            if self.parallel_mode == 'ddp' and self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            # 周期性采样快照（数据集可用 snapshot_allowed 声明此阶段不允许采样）
            if self.step % self.i_sample == 0 and snapshot_allowed(self.dataset):
                self.snapshot()

            if self.is_master:
                self.log.append((self.step, {}))

                # Log time
                self.log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    self.log[-1][1].update(step_log)

                # Log scale
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == 'amp':
                        self.log[-1][1]['scale'] = self.scaler.get_scale()
                    elif self.mix_precision_mode == 'inflat_all':
                        self.log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    self.save_logs()

                # Save checkpoint
                # 周期性保存完整 checkpoint 组（仅 rank0）
                if self.step % self.i_save == 0:
                    self.save()
                    
            # Check abort
            # 发散检查（可能先存盘再抛错中止）
            self.check_abort()

        # A non-i_save-aligned max_steps must still have a committed final
        # 中文：max_steps 未对齐 i_save 时也必须提交最终 checkpoint；已有 marker 则只校验、不覆盖旧步。
        # checkpoint. Never overwrite an old/partial step; helper fails closed.
        if self.is_master and getattr(self, '_last_committed_step', None) != self.step:
            ckpt_dir = os.path.join(self.output_dir, 'ckpts')
            if os.path.exists(os.path.join(ckpt_dir, marker_name(self.step))):
                validate_checkpoint(ckpt_dir, self.step, verify_hashes=True)
            else:
                self.save()
        # 收尾快照（受数据集的 snapshot_allowed 契约控制）
        if snapshot_allowed(self.dataset, final=True):
            self.snapshot(suffix='final', batch_size=self.snapshot_batch_size)
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        用 torch.profiler 采样若干步的开销，trace 写到 output_dir/profile。
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
