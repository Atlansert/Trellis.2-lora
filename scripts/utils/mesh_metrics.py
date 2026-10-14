#!/usr/bin/env python3
"""对推理产出的 GLB 计算客观几何指标，用于比较"细节/边缘流畅度"。

全部用 numpy 向量化实现（不依赖 trimesh 的 face_adjacency / split，
在 ~1M 面的网格上快很多）：

  sharp_edge_frac : 二面角 > 30° 的内部边占比 —— 越大说明表面越"碎/多折面"
  dihedral_mean   : 相邻面法向夹角均值（度）—— 越小越平滑
  dihedral_p95    : 95 分位夹角
  boundary_frac   : 边界边（仅 1 个相邻面）占比 —— 破洞/开口的直接体现
  n_components    : 顶点连通分量数 —— 破洞会把网格切成多块
  watertight      : 是否所有边都恰好有 2 个相邻面
"""
import os
import sys
import glob
import json
import argparse

import numpy as np


def load_geometry(path):
    import trimesh
    try:
        m = trimesh.load(path, force='mesh', process=False, skip_materials=True)
    except TypeError:
        m = trimesh.load(path, force='mesh', process=False)
    if hasattr(m, 'geometry'):          # Scene
        vs, fs, off = [], [], 0
        for g in m.geometry.values():
            vs.append(np.asarray(g.vertices, dtype=np.float64))
            fs.append(np.asarray(g.faces, dtype=np.int64) + off)
            off += len(vs[-1])
        return np.concatenate(vs), np.concatenate(fs)
    return (np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces, dtype=np.int64))


def metrics(path):
    v, f = load_geometry(path)
    out = {'file': os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path),
           'n_vertices': int(len(v)), 'n_faces': int(len(f))}

    # 面法向
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln

    # 无向边 → 相邻面
    corners = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1).reshape(-1, 2)
    e = np.sort(corners, axis=1)
    uniq, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    face_of_corner = np.repeat(np.arange(len(f), dtype=np.int64), 3)

    out['n_edges'] = int(len(uniq))
    out['boundary_edges'] = int((cnt == 1).sum())
    out['boundary_frac'] = float((cnt == 1).mean())
    out['nonmanifold_edges'] = int((cnt > 2).sum())
    out['watertight'] = bool((cnt == 2).all())

    two = np.nonzero(cnt == 2)[0]
    if len(two):
        order = np.argsort(inv, kind='stable')
        inv_s, face_s = inv[order], face_of_corner[order]
        st = np.searchsorted(inv_s, two)
        f1, f2 = face_s[st], face_s[st + 1]
        dot = np.clip(np.abs((n[f1] * n[f2]).sum(axis=1)), 0.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        out['sharp_edge_frac'] = float((ang > 30).mean())
        out['dihedral_mean'] = float(ang.mean())
        out['dihedral_p95'] = float(np.percentile(ang, 95))
        out['dihedral_p99'] = float(np.percentile(ang, 99))
    else:
        out.update(sharp_edge_frac=None, dihedral_mean=None,
                   dihedral_p95=None, dihedral_p99=None)

    # 顶点连通分量（用边连成的图）
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        A = coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])),
                       shape=(len(v), len(v)))
        ncomp, _ = connected_components(A, directed=False)
        sizes = np.bincount(np.arange(len(v)))  # 占位
        out['n_components'] = int(ncomp)
    except Exception as ex:
        out['n_components'] = None
        out['err_comp'] = str(ex)

    # 面积
    out['area'] = float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    out['bbox'] = [round(float(x), 4) for x in (v.max(0) - v.min(0))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glbs', nargs='+')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    paths = []
    for g in a.glbs:
        if os.path.isdir(g):
            paths += sorted(glob.glob(os.path.join(g, '*.glb')))
        else:
            paths += sorted(glob.glob(g))
    rows = []
    for p in paths:
        try:
            r = metrics(p)
            rows.append(r)
            print(f'  [完成] {r["file"]}', flush=True)
        except Exception as ex:
            print(f'  [失败] {p}: {type(ex).__name__}: {ex}', flush=True)
    if not rows:
        print('无可用 GLB'); return 1
    print()
    print(f'{"实验":<44}{"顶点":>9}{"面":>9}{"尖边占比":>10}{"夹角均值":>10}'
          f'{"夹角p95":>9}{"边界边占比":>12}{"分量":>7}{"封闭":>7}')
    for r in rows:
        def f3(k, w=3):
            x = r.get(k)
            return f'{x:>{w}.{3 if w>=3 else 2}f}' if isinstance(x, float) else f'{str(x):>{w}}'
        print(f'{r["file"][:44]:<44}{r["n_vertices"]:>9}{r["n_faces"]:>9}'
              f'{f3("sharp_edge_frac",10)}{f3("dihedral_mean",10)}'
              f'{f3("dihedral_p95",9)}{f3("boundary_frac",12)}'
              f'{str(r["n_components"]):>7}{str(r["watertight"]):>7}')
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=2, ensure_ascii=False)
        print(f'\n已写出 {a.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())