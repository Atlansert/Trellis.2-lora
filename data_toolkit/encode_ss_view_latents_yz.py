#!/usr/bin/env python3
"""生成"已做 y/z 对齐"的逐视角 SS latent（治本修复，替代推理侧的 --ss_swap_yz）。

背景（实测）：本地逐视角 SS 体素化脚本直接使用原始 mesh dump 坐标，缺少
blender_dump ↔ dual_grid 的 y/z 对齐（HANDOFF §6.1-4 只为 PBR/shape 管线做了）。
结果本地 SS 训练目标与预训练模型/管线约定整整差一次 y/z 交换：
    IoU(base, GT) = 0.026   →   IoU(swap_yz(base), GT) = 0.217

体素化是逐轴独立的，因此在**体素坐标层**交换 y/z 与"先交换网格轴再体素化"等价，
无需重跑 open3d 体素化，只要重新编码即可。

输出：<root>/<output_name>/<obj>/NNN.npz，键名与既有目录一致（'mean'），
可直接换数据集的 ss_view_latent 指向使用。

中文说明：
- 用途：生成“已做 y/z 对齐”的逐视角 SS latent，从数据层治本修复，替代推理侧的 --ss_swap_yz。
- 输入：<root>/<voxel_name>/<obj>/NNN.npz 的 coords；--encoder 为 SS 编码器（需要 GPU）。
- 输出：<root>/<output_name>/<obj>/NNN.npz，键 'mean'，可直接替换数据集的 ss_view_latent 指向。
- 关键约定：体素化逐轴独立，所以在**体素坐标层**交换 y/z（[0,2,1]）等价于先交换网格轴再体素化，
  无需重跑 open3d；默认开启且不可通过命令行关闭（--swap_yz 默认 True）。
- 用法：python encode_ss_view_latents_yz.py --root <dir> [--voxel_name ...] [--output_name ...]
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trellis2.models as models  # noqa: E402


# CLI 入口：逐批交换 y/z → 栅格化 → SS 编码 → 写 'mean'。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--voxel_name', default='ss_view_voxels_mesh_t1')
    ap.add_argument('--output_name', default='ss_latents_views_mesh_t1_yz')
    ap.add_argument('--encoder',
                    default='/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--resolution', type=int, default=64)
    ap.add_argument('--swap_yz', action='store_true', default=True,
                    help='交换体素坐标的 y/z（默认开启；这是本脚本的目的）')
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    inp = os.path.join(root, a.voxel_name)
    out = os.path.join(root, a.output_name)
    os.makedirs(out, exist_ok=True)

    enc = models.from_pretrained(a.encoder).eval().cuda()
    files = sorted(glob.glob(os.path.join(inp, '*', '*.npz')))
    print(f'files={len(files)} swap_yz={a.swap_yz} encoder={a.encoder}', flush=True)

    res = a.resolution
    with torch.no_grad():
        for st in tqdm(range(0, len(files), a.batch_size), desc='Encoding aligned latents'):
            batch = files[st:st + a.batch_size]
            grids = []
            for p in batch:
                c = np.load(p)['coords'].astype(np.int64)
                if a.swap_yz:
                    c = c[:, [0, 2, 1]]          # 交换 y/z
                c = np.clip(c, 0, res - 1)
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