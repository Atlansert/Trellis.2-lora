#!/usr/bin/env python3
"""判定"旧配置是否真的没有朝向反转问题"。

思路：旧 run 一直被使用的只是 33k EMA。但它其实有 10k/20k/.../50k 的 EMA，
以及 50k 的**原始权重**（未做 EMA）。如果旧配置本身没病，那么旧 50k 原始权重
（LoRA 完全生效）也应当稳定输出 h-long；如果它也摇摆，说明旧配置同样有病，
33k 看着正常只是因为"训练不足 + 条件编码器错配"把 LoRA 压住了。

同时输出各配置占据体与 base 的 IoU，量化"有多接近 base"。
"""
import os
import sys
import json

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
P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
         guidance_interval=[0.6, 1.0], rescale_t=5.0)

OLD = 'results/multiview_1186_ss_flow_lora_64_views/ckpts'
NEW = 'results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts'
CKS = [
    ('旧 EMA 10k', f'{OLD}/denoiser_ema0.9999_step0010000.pt'),
    ('旧 EMA 33k', f'{OLD}/denoiser_ema0.9999_step0033000.pt'),
    ('旧 EMA 50k', f'{OLD}/denoiser_ema0.9999_step0050000.pt'),
    ('旧 原始 50k(无EMA)', f'{OLD}/denoiser_step0050000.pt'),
    ('新A EMA 50k', f'{NEW}/denoiser_ema0.9999_step0050000.pt'),
]
IMGS = ['001', '002', '003', '004']


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']

    imgs, cond3 = {}, {}
    for n in IMGS:
        im = pipe.preprocess_image(Image.open(f'data_test/{n}.png'))
        imgs[n] = im
        cond3[n] = pipe.get_ss_cond([im])

    m3 = pipe.ss_image_cond_model
    from trellis2.trainers.flow_matching.mixins.image_conditioned import (
        DinoV2LocalFeatureExtractor)
    pipe.ss_image_cond_model = DinoV2LocalFeatureExtractor(
        model_name='/data/lzx/ReconViaGen/checkpoints_dinov2', image_size=518)
    pipe.ss_cond_resolution, pipe.ss_cond_crop_scale = 518, 1.2
    cond2 = {n: pipe.get_ss_cond([imgs[n]]) for n in IMGS}
    pipe.ss_image_cond_model = m3

    def occ(cond):
        torch.manual_seed(42)
        c = pipe.sample_sparse_structure(cond, 64, 1, dict(P)).cpu().numpy()
        o = np.zeros((64, 64, 64), bool)
        for d, h, w in c[:, 1:]:
            o[d, h, w] = True
        return o

    def desc(o):
        nz = np.argwhere(o)
        if not len(nz):
            return 'empty', (0, 0, 0)
        ext = tuple(int(v) for v in (nz.max(0) - nz.min(0) + 1))
        return 'xyz'[int(np.argmax(ext))], ext

    def iou(a, b):
        u = (a | b).sum()
        return (a & b).sum() / u if u else 0.0

    base3 = {n: occ(cond3[n]) for n in IMGS}
    base2 = {n: occ(cond2[n]) for n in IMGS}

    def run_table(cond, bases, title):
        print(f'\n### 条件编码器 = {title}')
        print(f'{"SS 权重":<22}{"图":<5}{"长轴":<6}{"跨度(x,y,z)":<17}{"vs base IoU":>11}')
        print('-' * 64)
        for tag, ck in CKS:
            if not os.path.isfile(ck):
                print(f'{tag:<22}<检查点缺失 {ck}>')
                continue
            sd = torch.load(ck, map_location='cpu', weights_only=True)
            sd.pop('rope_phases', None)
            apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
            r = ss.load_state_dict(sd, strict=False)
            for n in IMGS:
                o = occ(cond[n])
                ax, ext = desc(o)
                flag = '' if ax in ('h' if False else 'y',) else ''
                print(f'{tag:<22}{n:<5}{ax:<6}{str(ext):<17}{iou(o, bases[n]):>11.3f}')
            print()

    run_table(cond3, base3, 'DINOv3@512（= 推理管线默认，新 run 的训练口径）')
    run_table(cond2, base2, 'DINOv2@518+1.2（= 旧 run 的训练口径）')
    return 0


if __name__ == '__main__':
    sys.exit(main())