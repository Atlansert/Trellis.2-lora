#!/usr/bin/env python3
"""用原厂 base 模型标定：修正后的逐视角体素 GT 应该对网格轴做哪种置换，
才能落到预训练模型/推理管线使用的坐标系上。

做法：对若干训练对象、若干视角，用 base（不挂 LoRA）在对应渲染图上跑 SS 得到
占据体 P，再拿修正后的 GT 占据体 G，遍历 6 种轴置换计算 IoU，取均值最高者。

只有循环置换（3-cycle）才是纯旋转；对换会引入镜像。两种都测，用于判断是否存在
手性差异。
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

P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
         guidance_interval=[0.6, 1.0], rescale_t=5.0)
ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon'
N_OBJ = 16
VIEWS = [0, 6, 12, 18]


def main():
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()

    objs = sorted(os.listdir(f'{ROOT}/{VOX}'))[:N_OBJ]
    perms = list(itertools.permutations(range(3)))

    def occ_gt(path):
        c = np.load(path)['coords'].astype(np.int64)
        o = np.zeros((64, 64, 64), bool)
        c = np.clip(c, 0, 63)
        if len(c):
            o[c[:, 0], c[:, 1], c[:, 2]] = True
        return o

    def permute(o, p):
        return np.transpose(o, p)

    def iou(a, b):
        u = (a | b).sum()
        return float((a & b).sum()) / u if u else 0.0

    acc = {p: [] for p in perms}
    n = 0
    for o in objs:
        for v in VIEWS:
            gp = f'{ROOT}/{VOX}/{o}/{v:03d}.npz'
            rp = glob.glob(f'{ROOT}/renders_cond/{o}/*.png')
            rp = sorted(rp)
            if not os.path.isfile(gp) or v >= len(rp):
                continue
            img = pipe.preprocess_image(Image.open(rp[v]))
            torch.manual_seed(42)
            c = pipe.sample_sparse_structure(pipe.get_ss_cond([img]), 64, 1, dict(P)).cpu().numpy()
            pb = np.zeros((64, 64, 64), bool)
            for d, h, w in c[:, 1:]:
                pb[d, h, w] = True
            g = occ_gt(gp)
            for p in perms:
                acc[p].append(iou(pb, permute(g, p)))
            n += 1
        print(f'{o}: {n} 对', flush=True)

    print(f'\n样本对数 = {n}')
    print(f'{"置换 (E0,E1,E2)->":<22}{"平均IoU":>9}{"中位":>8}  类型')
    print('-' * 56)
    for p in sorted(perms, key=lambda p: -np.mean(acc[p])):
        cyc = '循环(纯旋转)' if p in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else '对换(镜像)'
        print(f'{str(p):<22}{np.mean(acc[p]):>9.3f}{np.median(acc[p]):>8.3f}  {cyc}')
    return 0


if __name__ == '__main__':
    sys.exit(main())