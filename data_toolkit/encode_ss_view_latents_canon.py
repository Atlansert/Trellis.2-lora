#!/usr/bin/env python3
"""把**修正相机约定后**的逐视角体素编码成 SS latent，并对齐到预训练/推理管线约定。

输入 : <root>/ss_view_voxels_canon_v4（由 generate_ss_view_voxels_canon.py 生成）
输出 : <root>/<output_name>/<obj>/NNN.npz，键 'mean'（与既有目录一致）

两步：
1. 轴置换 (2,0,1)：修正后的网格轴语义是 (画面水平, 视深, 画面竖直)，
   而预训练模型/管线约定是 (竖直, 长度, 视深) —— 依据是原厂 base 在同类渲染上
   输出的跨度形如 (10, 64, 10)（长度在 index1）。该置换是循环置换（纯旋转），
   不会破坏上一步已修好的手性。
2. 用 SS VAE（TRELLIS.1 的 ss_enc_conv3d_16l8_fp16）编码，`sample_posterior=False`。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trellis2.models as models  # noqa: E402

# 修正后网格 -> 管线约定：新[i] = 旧[PERM[i]]
PERM = (2, 0, 1)


# CLI 入口：读 --voxel_name 下的逐视角体素，按 --perm 置换轴后编码，写到 --output_name/<obj>/NNN.npz。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--voxel_name', default='ss_view_voxels_canon_v4')
    ap.add_argument('--output_name', default='ss_latents_views_canon')
    ap.add_argument('--encoder',
                    default='/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--resolution', type=int, default=64)
    ap.add_argument('--perm', default='2,0,1',
                    help='轴置换，默认标定值 "2,0,1"；传 "none" 表示不做')
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    inp = os.path.join(root, a.voxel_name)
    out = os.path.join(root, a.output_name)
    os.makedirs(out, exist_ok=True)
    perm = None if a.perm == 'none' else [int(x) for x in a.perm.split(',')]

    enc = models.from_pretrained(a.encoder).eval()
    enc.to('cuda')          # 注意：不要用 .cuda()，某些实现不返回 self
    assert next(enc.parameters()).is_cuda, 'encoder 未搬到 GPU'

    files = sorted(glob.glob(os.path.join(inp, '*', '*.npz')))
    print(f'files={len(files)} perm={perm} encoder={a.encoder}', flush=True)

    res = a.resolution
    with torch.no_grad():
        for st in tqdm(range(0, len(files), a.batch_size), desc='Encoding canon latents'):
            batch = files[st:st + a.batch_size]
            grids = []
            for p in batch:
                c = np.load(p)['coords'].astype(np.int64)
                if perm is not None and len(c):
                    c = c[:, perm]
                c = np.clip(c, 0, res - 1)
                # 把稀疏整数坐标栅格化为 res^3 的二值占据网格（值 1 表示占据）。
                x = torch.zeros(res, res, res, dtype=torch.float32)
                if len(c):
                    x[c[:, 0], c[:, 1], c[:, 2]] = 1
                grids.append(x)
            x = torch.stack(grids, 0)[:, None].cuda()
            z = enc(x, sample_posterior=False).cpu().numpy()
            for p, zz in zip(batch, z):
                obj = os.path.basename(os.path.dirname(p))
                od = os.path.join(out, obj)
                os.makedirs(od, exist_ok=True)
                np.savez_compressed(os.path.join(od, os.path.basename(p)), mean=zz)
    print('done', flush=True)


if __name__ == '__main__':
    main()