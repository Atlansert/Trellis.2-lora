#!/usr/bin/env python3
"""直接搜索正确的体素化变换：用"GT 剪影 vs 渲染剪影"的 IoU 打分。

候选空间（由前面的实测约束缩小）：
  M     : dump->渲染帧的轴映射，(x, z, y) 的 4 种符号变体（|extents| 不变，
          因此都能通过宽高比检验，但手性/视深符号不同）
  offset: 环绕角偏移 270 或 90（差 180°，即绕竖轴多转半圈，决定画面左右是否颠倒）

评分：把 GT 占据体沿视深轴投影成剪影，与渲染图 alpha 剪影（按前景 bbox 归一化）
算 IoU；同时算"GT 左右翻转"后的 IoU。正确候选应当**未翻转的 IoU 最高**。
"""
import os
import sys
import glob
import itertools

os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = 'training_data/multiview_1186'
OBJS = ['0086bbab37559442', '00989799ab60309b', '008ae427b2606cc8',
        '015de7a65b71fbc2', '01ee1705027b3e0f']
VIEWS = [0, 4, 8, 12, 16, 20]
RES = 64


def rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def mesh_of(o):
    import pickle
    d = pickle.load(open(f'{ROOT}/mesh_dumps/{o}.pickle', 'rb'))
    V, F, off = [], [], 0
    for p in d['objects']:
        v = np.asarray(p['vertices'])
        f = np.asarray(p['faces'])
        if len(v):
            V.append(v); F.append(f.astype(np.int64) + off); off += len(v)
    return np.concatenate(V).astype(float), np.concatenate(F)


def vox(V, F, yaw, M, offset):
    vv = (V * np.array(M)) @ rz(yaw - offset)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),
                                  o3d.utility.Vector3iVector(F))
    g = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        m, voxel_size=1 / RES, min_bound=(-.5, -.5, -.5), max_bound=(.5, .5, .5))
    c = np.array([v.grid_index for v in g.get_voxels()])
    occ = np.zeros((RES, RES, RES), bool)
    if len(c):
        c = np.clip(c, 0, RES - 1)
        occ[c[:, 0], c[:, 1], c[:, 2]] = True
    return occ


def crop_norm(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return np.zeros((RES, RES), bool)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((RES, RES), Image.BOX)) > 127


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def main():
    renders = {}
    for o in OBJS:
        for v in VIEWS:
            fp = sorted(glob.glob(f'{ROOT}/renders_cond_t/{o}/*.png'))[v]
            a = np.array(Image.open(fp).convert('RGBA').getchannel('A')) > 128
            renders[(o, v)] = crop_norm(a.T)          # (x, y) 与 GT 投影同序

    cands = []
    for perm, signs in itertools.product([(0, 2, 1), (2, 0, 1)], itertools.product([1, -1], repeat=3)):
        for offset in (270.0, 90.0):
            cands.append((perm, signs, offset))

    print(f'{"置换":<10}{"符号":<16}{"偏移":>6}{"未翻转IoU":>11}{"翻转IoU":>10}{"差值":>8}')
    print('-' * 64)
    rows = []
    meshes = {o: mesh_of(o) for o in OBJS}
    for perm, signs, offset in cands:
        V, F = meshes[OBJS[0]]
        acc, accf = [], []
        for o in OBJS:
            V, F = meshes[o]
            Vm = V[:, list(perm)] * np.array(signs)
            for v in VIEWS:
                occ = vox(Vm, F, v * 15.0, (1, 1, 1), offset)
                proj = occ.any(axis=1)
                if proj.sum() < 4:
                    continue
                P = crop_norm(proj)
                acc.append(iou(P, renders[(o, v)]))
                accf.append(iou(crop_norm(np.fliplr(proj)), renders[(o, v)]))
        rows.append((np.mean(acc), np.mean(accf), perm, signs, offset))
    for a, af, perm, signs, offset in sorted(rows, key=lambda r: -r[0]):
        print(f'{str(perm):<10}{str(signs):<16}{offset:>6.0f}{a:>11.3f}{af:>10.3f}{a - af:>+8.3f}')
    best = max(rows, key=lambda r: r[0])
    print(f'\n最佳 = 置换{best[2]} 符号{best[3]} 偏移{best[4]:.0f}  IoU={best[0]:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())