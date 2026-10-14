#!/usr/bin/env python3
"""用原厂 base 模型标定修正后 GT 需要的轴置换（解码后比较）。

关键修正：open3d 的 `create_from_triangle_mesh_within_bounds` 产出的是**表面壳**体素，
而 base 的 SS 输出是**实心**占据体，直接算 IoU 是苹果比橘子（实测最高只有 0.136）。
因此先把 GT 体素用 SS VAE 编码再解码成实心占据体，再与 base 比较。

对 6 种轴置换分别评估，取平均 IoU 最高者。只有 3-cycle 是纯旋转，对换会引入镜像。
"""
import os
import sys
import glob
import itertools

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import trellis2.models as models

P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
         guidance_interval=[0.6, 1.0], rescale_t=5.0)
ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon'
ENC = '/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16'
N_OBJ = 16
VIEWS = [0, 6, 12, 18]


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    dec = pipe.models['sparse_structure_decoder'].cuda()
    wdt = next(dec.parameters()).dtype
    enc = models.from_pretrained(ENC).eval().cuda()

    objs = sorted(os.listdir(f'{ROOT}/{VOX}'))[:N_OBJ]
    perms = list(itertools.permutations(range(3)))

    def gt_occ(path, perm):
        c = np.load(path)['coords'].astype(np.int64)
        if len(c):
            c = c[:, perm]
        x = torch.zeros(64, 64, 64, dtype=torch.float32)
        if len(c):
            x[c[:, 0], c[:, 1], c[:, 2]] = 1
        with torch.no_grad():
            z = enc(x[None, None].cuda(), sample_posterior=False)
            o = (dec(z.to(wdt)) > 0)[0, 0].cpu().numpy()
        return o

    def iou(a, b):
        u = (a | b).sum()
        return float((a & b).sum()) / u if u else 0.0

    acc = {p: [] for p in perms}
    n = 0
    for o in objs:
        rp = sorted(glob.glob(f'{ROOT}/renders_cond/{o}/*.png'))
        for v in VIEWS:
            gp = f'{ROOT}/{VOX}/{o}/{v:03d}.npz'
            if not os.path.isfile(gp) or v >= len(rp):
                continue
            img = pipe.preprocess_image(Image.open(rp[v]))
            torch.manual_seed(42)
            c = pipe.sample_sparse_structure(pipe.get_ss_cond([img]), 64, 1, dict(P)).cpu().numpy()
            pb = np.zeros((64, 64, 64), bool)
            for d, h, w in c[:, 1:]:
                pb[d, h, w] = True
            for p in perms:
                acc[p].append(iou(pb, gt_occ(gp, p)))
            n += 1
        print(f'{o}: {n} 对', flush=True)

    print(f'\n样本对数 = {n}')
    print(f'{"置换":<14}{"平均IoU":>9}{"中位":>8}  类型')
    print('-' * 50)
    for p in sorted(perms, key=lambda p: -np.mean(acc[p])):
        cyc = '循环(纯旋转)' if p in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else '对换(镜像)'
        print(f'{str(p):<14}{np.mean(acc[p]):>9.3f}{np.median(acc[p]):>8.3f}  {cyc}')
    return 0


if __name__ == '__main__':
    sys.exit(main())