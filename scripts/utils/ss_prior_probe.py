#!/usr/bin/env python3
"""判定：SS 输出的朝向是"模型先验"还是"被图像条件拉过去的"。

做法：把条件置零（cond = neg_cond = 0）做纯无条件采样，看各权重输出的长轴。
若各权重在无条件下都落在同一根轴，说明"正确朝向"是模型先验，
反转是图像条件把它拉走的——那么新配置的毛病就是"条件通路被训得更"强/更歪"。
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
CKS = [('base 无LoRA', None),
       ('旧 EMA 50k', f'{OLD}/denoiser_ema0.9999_step0050000.pt'),
       ('新A EMA 50k', f'{NEW}/denoiser_ema0.9999_step0050000.pt')]
IMGS = ['001', '002', '003', '004']


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']

    conds = {}
    for n in IMGS:
        im = pipe.preprocess_image(Image.open(f'data_test/{n}.png'))
        conds[n] = pipe.get_ss_cond([im])

    ss_c = pipe.models['sparse_structure_flow_model'].in_channels
    # 条件特征维度取自实际 cond
    k = next(iter(conds.values()))['cond']
    zero = {'cond': torch.zeros_like(k), 'neg_cond': torch.zeros_like(k)}

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
            return 'empty', (0, 0, 0), 0
        ext = tuple(int(v) for v in (nz.max(0) - nz.min(0) + 1))
        return 'xyz'[int(np.argmax(ext))], ext, int(o.sum())

    print(f'{"权重":<14}{"条件":<10}{"图":<5}{"长轴":<6}{"跨度(x,y,z)":<17}{"体素":>6}')
    print('-' * 62)
    for tag, ck in CKS:
        if ck is not None:
            sd = torch.load(ck, map_location='cpu', weights_only=True)
            sd.pop('rope_phases', None)
            apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
            ss.load_state_dict(sd, strict=False)
        for mode, cdict in [('无条件(零)', zero)] + [('图像条件', None)]:
            for n in IMGS:
                c = zero if cdict is not None else conds[n]
                ax, ext, nv = desc(occ(c))
                print(f'{tag:<14}{mode:<10}{n:<5}{ax:<6}{str(ext):<17}{nv:>6}')
            print()
    return 0


if __name__ == '__main__':
    sys.exit(main())