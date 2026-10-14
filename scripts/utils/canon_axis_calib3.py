#!/usr/bin/env python3
"""用原厂 base 标定修正后 GT 的轴置换（填洞版）。

open3d 体素化产出**表面壳**，base 的 SS 输出是**实心**，直接比 IoU 上限很低。
这里用 scipy.ndimage.binary_fill_holes 把 GT 壳填成实心后再比，信号干净得多。

运行：无参数；需 GPU（默认卡 5），脚本内已 chdir 到项目根，ROOT 等为相对路径常量。
安全边界：只读 GT npz、renders_cond 图与 base 权重，不写任何文件；纯标定实验。
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
N_OBJ = 20
VIEWS = [0, 4, 8, 12, 16, 20]


def main():
    """对若干对象/视角：base 采样出占据体，与"按置换 p 摆放并填洞"的 GT 比 IoU；6 种置换取均值最高者为解。"""
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    objs = sorted(os.listdir(f'{ROOT}/{VOX}'))[:N_OBJ]
    perms = list(itertools.permutations(range(3)))

    def gt_solid(path, perm):
        """读 GT 体素，按置换 perm 重排轴并填洞成实心（这样才和 base 的实心占据体可比）。"""
        c = np.load(path)['coords'].astype(np.int64)
        x = np.zeros((64, 64, 64), bool)
        if len(c):
            c = np.clip(c[:, perm], 0, 63)
            x[c[:, 0], c[:, 1], c[:, 2]] = True
        return ndimage.binary_fill_holes(x)

    def iou(a, b):
        """体素 IoU；两者都空时定义为 0。"""
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
                acc[p].append(iou(pb, gt_solid(gp, p)))
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