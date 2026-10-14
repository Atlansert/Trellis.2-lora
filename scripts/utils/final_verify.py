#!/usr/bin/env python3
"""最终验收：对若干 results_infer 目录，逐图给出
  成品 GLB 长度轴（判据：应为 z） + 几何质量（二面角均值/p95、顶点/面数、封闭性）

用法:
  final_verify.py tag=dir [tag=dir ...]

输出：stdout 对照表（tag=results_infer 子目录 × 默认 001..004 每图一行：长度轴/bbox/点数/面数/二面角）。
安全边界：只读 GLB，不写任何文件；纯 CPU/trimesh，不改动实验产物；缺图或读取失败只打印标记不中断。
"""
import argparse
import glob
import os
import sys

import numpy as np


def metrics(path):
    """读 GLB 的 bbox/点数/面数，并只对"恰好被 2 个面共享"的边统计二面角均值与 p95；无面时只返回基础信息。"""
    import trimesh
    m = trimesh.load(path, force='mesh', process=False, skip_materials=True)
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    out = {'bbox': v.max(0) - v.min(0), 'nv': len(v), 'nf': len(f)}
    if len(f) == 0:
        return out
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    corners = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1).reshape(-1, 2)
    e = np.sort(corners, axis=1)
    uniq, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    out['watertight'] = bool((cnt == 2).all())
    two = np.nonzero(cnt == 2)[0]
    if len(two):
        order = np.argsort(inv, kind='stable')
        inv_s = inv[order]
        face_of_corner = np.repeat(np.arange(len(f), dtype=np.int64), 3)
        face_s = face_of_corner[order]
        st = np.searchsorted(inv_s, two)
        f1, f2 = face_s[st], face_s[st + 1]
        dot = np.clip(np.abs((n[f1] * n[f2]).sum(axis=1)), 0.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        out['dihedral_mean'] = float(ang.mean())
        out['dihedral_p95'] = float(np.percentile(ang, 95))
    return out


def main():
    """按 tag=目录 逐个实验、逐图输出朝向与几何质量对照表（长度轴为 z 视为规范）；返回 0。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', help='tag=results_infer/<子目录>')
    ap.add_argument('--images', nargs='+', default=['001', '002', '003', '004'])
    a = ap.parse_args()

    print(f'{"实验":<30}{"图":<6}{"长度轴":<8}{"bbox":<24}{"顶点":>9}{"面":>9}'
          f'{"二面角均值↓":>12}{"p95↓":>8}')
    print('-' * 112)
    for spec in a.dirs:
        tag, d = spec.split('=', 1)
        for n in a.images:
            p = os.path.join(d, f'{n}.glb')
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                print(f'{tag:<30}{n:<6}{"<缺失>":<8}')
                continue
            try:
                m = metrics(p)
            except Exception as ex:
                print(f'{tag:<30}{n:<6}<读取失败 {ex}>')
                continue
            ax = 'xyz'[int(np.argmax(m['bbox']))]
            mark = 'z ✔' if ax == 'z' else f'{ax} ✘'
            bb = '(' + ','.join(f'{x:.2f}' for x in m['bbox']) + ')'
            dm = f"{m.get('dihedral_mean', float('nan')):.3f}"
            dp = f"{m.get('dihedral_p95', float('nan')):.2f}"
            print(f'{tag:<30}{n:<6}{mark:<8}{bb:<24}{m["nv"]:>9}{m["nf"]:>9}{dm:>12}{dp:>8}')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())