#!/usr/bin/env python3
"""用 base 的 3D 输出定论 GT 的**符号/手性**（剪影测不出，因为投影看不见视深符号）。

做法：把 base 的输出取"壳"（occupancy & ~erode(occupancy)），与修正后 GT 的壳
在 3D 上比 IoU —— 这样是壳比壳，不受"壳 vs 实心"的影响。
对轴置换 p=(2,0,1)（已由剪影/跨度确认为唯一循环解）逐一试 8 种符号组合。
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
from scipy import ndimage

from trellis2.pipelines import Trellis2ImageTo3DPipeline

P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
         guidance_interval=[0.6, 1.0], rescale_t=5.0)
ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon'
N_OBJ = 24
VIEWS = [0, 4, 8, 12, 16, 20]
PERM = (2, 0, 1)
SIGNS = list(itertools.product([1, -1], repeat=3))


def shell(o):
    return o & ~ndimage.binary_erosion(o)


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    objs = sorted(os.listdir(f'{ROOT}/{VOX}'))[:N_OBJ]
    acc = {s: [] for s in SIGNS}

    def gt_shell(path, s):
        c = np.load(path)['coords'].astype(np.int64)
        c = c[:, PERM]
        c = (c * np.array(s) + (np.array(s) < 0) * 63)
        x = np.zeros((64, 64, 64), bool)
        x[np.clip(c[:, 0], 0, 63), np.clip(c[:, 1], 0, 63), np.clip(c[:, 2], 0, 63)] = True
        return x

    def iou(a, b):
        u = (a | b).sum()
        return float((a & b).sum()) / u if u else 0.0

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
            ps = shell(pb)
            for s in SIGNS:
                acc[s].append(iou(ps, gt_shell(gp, s)))
            n += 1
        print(f'{o}: {n} 对', flush=True)

    print(f'\n样本对数 = {n}   轴置换 = {PERM}')
    print(f'{"符号(s0,s1,s2)":<18}{"平均IoU":>9}{"中位":>8}  行列式')
    print('-' * 52)
    for s in sorted(SIGNS, key=lambda s: -np.mean(acc[s])):
        det = s[0] * s[1] * s[2]
        print(f'{str(s):<18}{np.mean(acc[s]):>9.3f}{np.median(acc[s]):>8.3f}  {"+1 纯旋转" if det > 0 else "-1 镜像"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())