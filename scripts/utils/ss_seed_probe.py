#!/usr/bin/env python3
"""判定"翻转"是确定性错误还是采样抖动。

对同一张图只换随机种子重复采样：若长轴在同一张图内就随种子变化，说明输出朝向
是欠定的（弱条件 / 没收敛），"翻转"本质是采样噪声；若同图内稳定、只在图与图之间
变化，才是真正学到的"随输入切换坐标系"。
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
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
    ('base 无LoRA', None, 'v3'),
    ('旧33k + DINOv3(错配=线上)', f'{OLD}/denoiser_ema0.9999_step0033000.pt', 'v3'),
    ('旧33k + DINOv2(匹配)', f'{OLD}/denoiser_ema0.9999_step0033000.pt', 'v2'),
    ('新A50k + DINOv3(匹配)', f'{NEW}/denoiser_ema0.9999_step0050000.pt', 'v3'),
    ('新A50k + DINOv2(错配)', f'{NEW}/denoiser_ema0.9999_step0050000.pt', 'v2'),
]
SEEDS = list(range(42, 52))
IMGS = ['001', '002', '003', '004']


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']

    raw = {n: Image.open(f'data_test/{n}.png') for n in IMGS}
    imgs = {n: pipe.preprocess_image(raw[n]) for n in IMGS}
    c3 = {n: pipe.get_ss_cond([imgs[n]]) for n in IMGS}

    m3 = pipe.ss_image_cond_model
    from trellis2.trainers.flow_matching.mixins.image_conditioned import (
        DinoV2LocalFeatureExtractor)
    pipe.ss_image_cond_model = DinoV2LocalFeatureExtractor(
        model_name='/data/lzx/ReconViaGen/checkpoints_dinov2', image_size=518)
    pipe.ss_cond_resolution, pipe.ss_cond_crop_scale = 518, 1.2
    c2 = {n: pipe.get_ss_cond([imgs[n]]) for n in IMGS}
    pipe.ss_image_cond_model = m3

    def occ(cond, seed):
        torch.manual_seed(seed)
        co = pipe.sample_sparse_structure(cond, 64, 1, dict(P)).cpu().numpy()
        o = np.zeros((64, 64, 64), bool)
        for d, h, w in co[:, 1:]:
            o[d, h, w] = True
        return o

    def ax(o):
        nz = np.argwhere(o)
        if not len(nz):
            return '-'
        ext = nz.max(0) - nz.min(0) + 1
        return 'xyz'[int(np.argmax(ext))]

    print(f'{"权重/条件":<24}{"图":<4}{"结论":<6}{"分布":<23}逐种子', flush=True)
    print('-'*72, flush=True)
    for tag, ck, ver in CKS:
        if ck is not None:
            sd = torch.load(ck, map_location='cpu', weights_only=True)
            sd.pop('rope_phases', None)
            apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
            ss.load_state_dict(sd, strict=False)
        cd = c3 if ver == 'v3' else c2
        for n in IMGS:
            axs = [ax(occ(cd[n], s)) for s in SEEDS]
            from collections import Counter
            c = Counter(axs); tot = sum(c.values())
            stable = '稳定' if len(c) == 1 else '摇摆'
            dist = ' '.join(f'{k}:{100*v/tot:.0f}%' for k, v in c.most_common())
            print(f'{tag:<24}{n}  {stable:<5} {dist:<22} {axs}', flush=True)
        print(flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())