#!/usr/bin/env python3
"""验证规范化后的逐视角 SS 监督是否"图文一致"。

判据：把每个视角的渲染剪影宽高比，与对应 latent（解码后占据体）的
"水平轴跨度 / 竖直轴跨度"比较。
  · 规范化正确 → 两者高度相关（渲染宽 ⇔ latent 水平跨度大），且
    latent 的长度轴随视角连续旋转（不再像旧数据那样对视角不变）。
  · 旧数据 → 占 71% 的对象 latent 与视角无关，相关系数接近 0。

latent 轴语义（已对齐到管线约定，轴置换 (2,0,1) 之后）：
  index0 = 竖直、index1 = 长度(画面水平)、index2 = 视深
"""
import argparse
import glob
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='training_data/multiview_1186')
    ap.add_argument('--latent', default='ss_latents_views_canon')
    ap.add_argument('--n_obj', type=int, default=40)
    ap.add_argument('--views', type=int, default=8)
    a = ap.parse_args()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    # 注意：必须先用管线自身的采样路径（会正确初始化稀疏卷积后端），
    # 再把解码器搬到 GPU；否则 Conv3d 会退化到 CUDA 不支持的 slow_conv3d。
    dec = pipe.models['sparse_structure_decoder']
    wdt = next(dec.parameters()).dtype
    # 备注：在独立脚本里把该解码器放 CUDA 会退化到 CUDA 不支持的 slow_conv3d
    # （管线内部路径能用，推测与稀疏卷积后端初始化有关）。这里直接在 CPU 解码，
    # 240 次 16^3->64^3 只需一两分钟。
    dec.to('cpu').eval()

    objs = sorted(os.listdir(os.path.join(a.root, a.latent)))[:a.n_obj]
    ars, hs = [], []
    per_obj_axis = []
    for o in objs:
        axes = []
        rp = sorted(glob.glob(os.path.join(a.root, 'renders_cond_t', o, '*.png')))
        lat = sorted(glob.glob(os.path.join(a.root, a.latent, o, '*.npz')))
        step = max(1, len(lat) // a.views)
        for i in range(0, len(lat), step):
            z = torch.from_numpy(np.load(lat[i])['mean']).float()
            occ = (dec(z[None].float()) > 0)[0, 0].numpy()
            nz = np.argwhere(occ)
            if not len(nz) or i >= len(rp):
                continue
            e = nz.max(0) - nz.min(0) + 1
            if e[1] < 4:
                continue
            # 渲染剪影宽高比
            im = np.array(Image.open(rp[i]).convert('RGBA').getchannel('A'))
            ys, xs = np.nonzero(im > 128)
            if len(xs) < 20:
                continue
            ar = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
            ars.append(ar)
            hs.append(e[1] / e[0])          # 长度轴 / 竖直轴
            axes.append('xyz'[int(np.argmax(e))])
        per_obj_axis.append(len(set(axes)))
    ars = np.array(ars)
    hs = np.array(hs)
    print(f'样本 = {len(ars)}  对象 = {len(objs)}')
    print(f'渲染宽高比 vs latent(E1/E0) 相关系数 = {np.corrcoef(ars, hs)[0,1]:.3f}')
    print(f'  渲染宽高比 中位 = {np.median(ars):.2f}   latent E1/E0 中位 = {np.median(hs):.2f}')
    uniform = sum(1 for k in per_obj_axis if k == 1)
    print(f'latent 长轴在各视角始终同一根轴的对象数 = {uniform}/{len(objs)}'
          f'  （旧数据 71% 的对象会是这个情况；规范化后应显著更低）')
    return 0


if __name__ == '__main__':
    sys.exit(main())