#!/usr/bin/env python3
"""扫描所有 sweep 产物，输出统一对照表（含步数、朝向、平滑度指标）。

朝向判据：读成品 GLB 的 bbox 最长轴，规范化帧下火车长度轴应为 z。

⚠️ 2026-10-11 更正：这里原先写"SS 输出期望长度轴 = z"是错的。实测下游把 SS 占据体
的**第 2 轴（h）**当长度轴。`--ss_swap_yz` 做的正是 h↔w 交换，因此它只是在给训歪的
模型打补丁；真正的根因是训练目标是混合帧，见 HANDOFF §6.1-13。
"""
import os
import re
import glob
import json
import argparse

import numpy as np


def bbox_of(p):
    import trimesh
    m = trimesh.load(p, force='mesh', process=False, skip_materials=True)
    return np.asarray(m.bounds[1] - m.bounds[0], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results_infer')
    ap.add_argument('--metrics-dir', default='/tmp')
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, 'sweep_*'))):
        g = os.path.join(d, '001.glb')
        if not os.path.isfile(g) or os.path.getsize(g) == 0:
            continue
        tag = os.path.basename(d)
        m = re.search(r'_(\d{7})_', tag)
        step = int(m.group(1)) if m else -1
        kind = 'B 规范帧' if 'canonical' in tag else 'A 视角帧'
        b = bbox_of(g)
        ax = 'xyz'[int(np.argmax(b))]
        # 指标文件
        mj = os.path.join(a.metrics_dir, f'metrics_{step}_{kind[0]}.json')
        mm = None
        if os.path.isfile(mj):
            try:
                mm = json.load(open(mj))[0]
            except Exception:
                mm = None
        rows.append(dict(kind=kind, step=step, bbox=b, ax=ax,
                         size=os.path.getsize(g), metrics=mm, dir=d))
    if not rows:
        print('无 sweep 产物'); return 1
    rows.sort(key=lambda r: (r['kind'], r['step']))
    print(f'{"组":<10}{"步数":>7}{"GLB":>9}{"bbox(x,y,z)":>26}{"长轴":>6}{"朝向":>8}'
          f'{"夹角均值":>10}{"夹角p95":>9}{"尖边占比":>10}')
    for r in rows:
        mm = r['metrics']
        dm = f'{mm["dihedral_mean"]:.3f}' if mm else '—'
        p95 = f'{mm["dihedral_p95"]:.2f}' if mm else '—'
        sf = f'{mm["sharp_edge_frac"]:.4f}' if mm else '—'
        ok = '✅规范' if r['ax'] == 'z' else '❌转90°'
        print(f'{r["kind"]:<10}{r["step"]:>7}{r["size"]/1e6:>8.1f}M'
              f'{str([round(x,3) for x in r["bbox"]]):>26}{r["ax"]:>6}{ok:>8}'
              f'{dm:>10}{p95:>9}{sf:>10}')
    print()
    print('说明：管线期望 SS 输出为规范帧（火车长度轴 = z）。')
    print('      长轴 = y 表示成品被绕轴转了 90°（视角帧监督的后果）。')
    print('      平滑度指标会被"退化成板"刷分，必须与朝向联读。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())