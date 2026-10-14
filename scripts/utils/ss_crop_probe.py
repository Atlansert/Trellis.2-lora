#!/usr/bin/env python3
"""隔离"旧 vs 新"的唯一剩余差异旋钮：条件图像的裁剪口径。

旧 run 训练时 crop_scale≈1.2（物体四周留边距），新 run 与推理默认都是 1.0（占满画面）。
条件编码器已确认是主因候选，这里固定 DINOv3@512，只改裁剪口径，看旧检查点是否
从"一致 h-long"变成"随输入摇摆"。
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
CKS = [('旧 EMA 50k', f'{OLD}/denoiser_ema0.9999_step0050000.pt'),
       ('新A EMA 50k', f'{NEW}/denoiser_ema0.9999_step0050000.pt')]
CROPS = [1.0, 1.2]
IMGS = ['001', '002', '003', '004']


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']

    raw = {n: Image.open(f'data_test/{n}.png') for n in IMGS}
    conds = {(n, c): pipe.get_ss_cond([pipe.preprocess_image(raw[n], crop_scale=c)])
             for n in IMGS for c in CROPS}

    def occ(cond):
        torch.manual_seed(42)
        co = pipe.sample_sparse_structure(cond, 64, 1, dict(P)).cpu().numpy()
        o = np.zeros((64, 64, 64), bool)
        for d, h, w in co[:, 1:]:
            o[d, h, w] = True
        return o

    def desc(o):
        nz = np.argwhere(o)
        if not len(nz):
            return 'empty'
        ext = nz.max(0) - nz.min(0) + 1
        return 'xyz'[int(np.argmax(ext))]

    print(f'{"SS 权重":<13}{"裁剪":<7}' + ''.join(f'{n:<6}' for n in IMGS) + '  一致?')
    print('-' * 56)
    for tag, ck in CKS:
        sd = torch.load(ck, map_location='cpu', weights_only=True)
        sd.pop('rope_phases', None)
        apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
        ss.load_state_dict(sd, strict=False)
        for c in CROPS:
            axs = [desc(occ(conds[(n, c)])) for n in IMGS]
            ok = '一致' if len(set(axs)) == 1 else f'摇摆({len(set(axs))} 种)'
            print(f'{tag:<13}{c:<7}' + ''.join(f'{a:<6}' for a in axs) + f'  {ok}')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())