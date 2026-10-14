"""训练启动入口。

负责：解析命令行与配置文件 → 分布式初始化 → 加载数据集与模型
→ 可选加载预训练权重 / 注入 LoRA → 构造 trainer 并运行。

安全要点：恢复/新建的合法性在写任何输出之前先校验；历史 config/command
只写一次不可覆盖；每次启动的尝试记录写入 output_dir/run_attempts，
失败重试只允许从本次输出目录中已提交的检查点恢复。
"""
import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random

from trellis2 import models, datasets, trainers
from trellis2.modules.lora import apply_lora_to_linear_layers
from trellis2.utils.dist_utils import setup_dist
from trellis2.utils.train_launch_contract import (
    resolve_checkpoint, preserve_launch_records, run_attempts, load_pretrained_checked)


def find_ckpt(cfg):
    # 恢复/新建的安全性校验，必须在任何输出写入之前完成
    return resolve_checkpoint(cfg)


def setup_rng(rank):
    # 各 rank 使用不同种子，避免所有进程采到相同随机数
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    # 生成逐参数的名称/形状/类型/是否训练表（完整表格会写入文件，控制台只打印统计）
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    # 多进程/多机时初始化通信组；单卡则跳过
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    # 按配置实例化数据集
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    # Build model
    # 按配置实例化各模型并放到 GPU
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    if hasattr(cfg, 'pretrained') and cfg.pretrained is not None:
        # 预训练权重按模型名匹配加载；safetensors 走 load_file，其余用 torch.load
        for name, path in cfg.pretrained.items():
            if name not in model_dict:
                raise ValueError(f'Pretrained target {name} not found in model_dict')
            if path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(path, device='cpu')
            else:
                state_dict = torch.load(path, map_location='cpu', weights_only=True)
            load_pretrained_checked(model_dict[name], state_dict)
            if rank == 0:
                print(f'[Pretrained] Loaded {name} from {path}')

    if hasattr(cfg, 'lora') and cfg.lora is not None and cfg.lora.get('enabled', False):
        # LoRA：先冻结原参数，再把目标线性层替换为 LoRA 版本；
        # 若一个可替换层都没匹配到，直接报错而不是静默跳过
        targets = cfg.lora.get('targets', [
            'self_attn.to_qkv', 'self_attn.to_out',
            'cross_attn.to_q', 'cross_attn.to_kv', 'cross_attn.to_out'])
        lora_models = cfg.lora.get('models', list(model_dict.keys()))
        for name, model in model_dict.items():
            if name not in lora_models:
                continue
            for param in model.parameters():
                param.requires_grad = False
            replaced = apply_lora_to_linear_layers(
                model, targets, rank=cfg.lora.get('rank', 8),
                alpha=cfg.lora.get('alpha', 8.0),
                dropout=cfg.lora.get('dropout', 0.0))
            if not replaced:
                raise RuntimeError(f'LoRA enabled for {name}, but no target layers matched')
            if rank == 0:
                print(f'[LoRA] Applied to {name}: {len(replaced)} layers')

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            # 只打印参数统计，完整表格写入文件
            num_total = sum(p.numel() for p in backbone.parameters())
            num_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            print(f'\n  Backbone: {name} | 总参数: {num_total:,} | 可训练: {num_trainable:,}')
            summary_root = os.path.join(cfg.output_dir, 'run_attempts')
            os.makedirs(summary_root, exist_ok=True)
            summary_file = os.path.join(summary_root, f'{cfg.launch_id}_attempt{cfg.attempt_index}_{name}_model_summary.txt')
            # 'x' 模式：文件已存在则报错，绝不覆盖历史尝试的记录
            with open(summary_file, 'x') as fp:
                print(model_summary, file=fp)

    # Build trainer
    # trainer 负责实际的训练循环与检查点写出
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)

    # Train
    # tryrun 只做一次前向检查，不进入训练
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == '__main__':
    # Arguments and config
    # 命令行参数 + 配置文件合并（配置文件可覆盖同名命令行项）
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=3, help='Maximum total attempts (historical compatibility): max(1, auto_retry); retries resume committed current-output checkpoints only')
    ## dubug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    with open(opt.config, 'r') as config_file:
        config = json.load(config_file)
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Validate resume/fresh safety BEFORE writing historical config/command.
    # 先校验恢复/新建是否安全，再写历史记录，避免污染已有输出目录
    cfg = find_ckpt(cfg)
    if cfg.num_gpus < 1:
        raise ValueError('Training requires at least one explicitly available GPU; no launch started')
    record_path = None
    record = None
    if cfg.node_rank == 0:
        import shlex
        # 只有 0 号节点写 config/command 与尝试记录；其它节点用随机 launch_id
        record_path, record = preserve_launch_records(
            cfg, config, shlex.join([sys.executable] + sys.argv), __file__)
        cfg.launch_id = record['launch_id']
    else:
        import uuid
        cfg.launch_id = 'node' + str(cfg.node_rank) + '_' + uuid.uuid4().hex
    def runner(current_cfg):
        # 记录本次是第几次尝试；多卡用 mp.spawn，单卡直接在当前进程跑
        current_cfg.attempt_index = len(record['attempts']) if record is not None else 1
        if current_cfg.num_gpus > 1:
            mp.spawn(main, args=(current_cfg,), nprocs=current_cfg.num_gpus, join=True)
        else:
            main(0, current_cfg)
    # Exceptions propagate through __main__, yielding nonzero exit.
    # run_attempts 负责按 auto_retry 重试，并且只从本次输出目录已提交的检查点恢复
    run_attempts(cfg, runner, record_path, record)
