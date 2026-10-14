#!/usr/bin/env python3
"""对 data_test/001..004 的真实照片，直接看 SS 阶段预测的占据体朝向。

目的：此前所有朝向结论都只建立在 001 一张图上（sweep 目录里只有 001.glb），
不足以支撑"某配置朝向稳定"的结论。本脚本跳过 shape/tex/remesh（那几步才是
分钟级的瓶颈），只跑 SS 采样 + 占据体解码，秒级给出：

    SS 占据体各轴跨度 → 长轴在哪根轴

约定：coords 为 (b,d,h,w)，d/h/w 对应管线规范帧的 x/y/z，火车长度轴应为 z。
若长轴落在 y 或 x，说明 SS 预测本身就把车转倒了——下游无法用全局变换救回。

用法: ss_orient_datatest.py [--images 001 002 003 004]
"""
import os
import sys
import argparse

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers

TARGETS = ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
           'cross_attn.to_kv', 'cross_attn.to_out']
P_INFER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=[0.6, 1.0], rescale_t=5.0)
SEED = 42

SS_CKS = [
    ('A 视角帧@50k(DINOv3训练)',
     'results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts/denoiser_ema0.9999_step0050000.pt'),
    ('A 视角帧@20k(DINOv3训练)',
     'results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts/denoiser_ema0.9999_step0020000.pt'),
    ('C 对齐版@50k',
     'results/multiview_1186_ss_flow_lora_64_views_alignedyz/ckpts/denoiser_ema0.9999_step0050000.pt'),
    ('C 对齐版@20k',
     'results/multiview_1186_ss_flow_lora_64_views_alignedyz/ckpts/denoiser_ema0.9999_step0020000.pt'),
    ('旧 33k(DINOv2训练)',
     'results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0033000.pt'),
]


def occ_of(coords, swap=False):
    c = coords.copy()
    if swap:
        c = c[:, [0, 1, 3, 2]]
    occ = np.zeros((64, 64, 64), bool)
    for d, h, w in c[:, 1:]:
        occ[d, h, w] = True
    return occ


def describe(occ):
    nz = np.argwhere(occ)
    if len(nz) == 0:
        return (0, (0, 0, 0), '-')
    ext = tuple(int(v) for v in (nz.max(0) - nz.min(0) + 1))
    return (int(occ.sum()), ext, 'xyz'[int(np.argmax(ext))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', default=['001', '002', '003', '004'])
    a = ap.parse_args()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']

    def sample(cond):
        torch.manual_seed(SEED)
        return pipe.sample_sparse_structure(cond, 64, 1, dict(P_INFER)).cpu().numpy()

    # 条件：管线默认（DINOv3）与 DINOv2 复刻口径
    conds = {}
    for name in a.images:
        img = pipe.preprocess_image(Image.open(f'data_test/{name}.png'))
        conds[name] = {'dino_v3(pipeline)': pipe.get_ss_cond([img]), '_img': img}

    from trellis2.trainers.flow_matching.mixins.image_conditioned import (
        DinoV2LocalFeatureExtractor)
    old_model = pipe.ss_image_cond_model
    old_res = getattr(pipe, 'ss_cond_resolution', 518)
    old_cs = getattr(pipe, 'ss_cond_crop_scale', 1.2)
    pipe.ss_image_cond_model = DinoV2LocalFeatureExtractor(
        model_name='/data/lzx/ReconViaGen/checkpoints_dinov2', image_size=518)
    pipe.ss_cond_resolution, pipe.ss_cond_crop_scale = 518, 1.2
    for name in a.images:
        conds[name]['dino_v2(旧口径)'] = pipe.get_ss_cond([conds[name]['_img']])
    pipe.ss_image_cond_model = old_model
    pipe.ss_cond_resolution, pipe.ss_cond_crop_scale = old_res, old_cs

    rows = []
    # base：先于 LoRA 注入
    for cname in ['dino_v3(pipeline)', 'dino_v2(旧口径)']:
        for name in a.images:
            occ = occ_of(sample(conds[name][cname]))
            rows.append((cname, 'base 无LoRA', name, *describe(occ)))

    apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
    for tag, ck in SS_CKS:
        if not os.path.isfile(ck):
            print(f'[跳过] 缺失 {tag}')
            continue
        sd = torch.load(ck, map_location='cpu', weights_only=True)
        sd.pop('rope_phases', None)
        miss, unexp = ss.load_state_dict(sd, strict=False)
        lora_keys = [k for k in sd if 'lora' in k.lower()]
        bn = float(sum(v.float().norm() for k, v in sd.items()
                       if 'lora' in k.lower() and v.dtype.is_floating_point))
        print(f'[载入] {tag}: sd键={len(sd)} LoRA键={len(lora_keys)} '
              f'missing={len(miss)} unexpected={len(unexp)} LoRA权重范数和={bn:.3f}')
        if miss:
            print('   missing 示例:', list(miss)[:4])
        for cname in ['dino_v3(pipeline)', 'dino_v2(旧口径)']:
            for name in a.images:
                occ = occ_of(sample(conds[name][cname]))
                rows.append((cname, tag, name, *describe(occ)))

    print('\n条件编码器        SS 权重               图   体素  跨度(x,y,z)      长轴')
    print('-' * 78)
    for c, t, n, nv, ext, ax in rows:
        flag = '  <-- 长度轴不是 z' if ax not in ('z', '-') else ''
        print(f'{c:<17} {t:<21} {n:<4} {nv:>5}  {str(ext):<16} {ax}{flag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())