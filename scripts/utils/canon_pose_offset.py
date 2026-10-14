#!/usr/bin/env python3
"""量化规范帧 latent 的"偏航姿态偏移"。

规范帧目标是按"渲染剪影最侧视"的视角挑出来的，而渲染相机是 15° 一档，
因此每个对象的规范帧姿态可能残留最多 ±7.5° 的偏航误差 —— 这会成为目标里的
又一处不一致，可能诱发翻转。

这里不重新体素化，而是用已有 24 个体素的侧视程度曲线
    r(v) = max(E0,E1) / E2          （E2 = 竖直，跨视角基本恒定）
做抛物线插值，估计真实峰值落在两个 15° 档之间的位置，
从而得到"所选档位与真实侧视方向"的偏差。

若偏差普遍接近 0，说明按档位挑选已足够；若普遍接近 ±7.5°，则应改为在
**精确侧视角**重新体素化规范帧。
"""
import glob
import os
import sys

import numpy as np

ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon_v4'


def main():
    objs = sorted(os.listdir(os.path.join(ROOT, VOX)))
    offs = []
    peak_ratio = []
    n_view = 24
    for o in objs:
        fs = sorted(glob.glob(os.path.join(ROOT, VOX, o, '*.npz')))
        if len(fs) != n_view:
            continue
        r = np.full(n_view, np.nan)
        for i, f in enumerate(fs):
            c = np.load(f)['coords'].astype(np.int64)
            if not len(c):
                continue
            e = c.max(0) - c.min(0) + 1
            r[i] = max(e[0], e[1]) / max(e[2], 1)
        if np.all(np.isnan(r)):
            continue
        k = int(np.nanargmax(r))
        # 抛物线插值（环绕）
        rm, r0, rp = r[(k - 1) % n_view], r[k], r[(k + 1) % n_view]
        den = rm - 2 * r0 + rp
        d = 0.5 * (rm - rp) / den if abs(den) > 1e-9 else 0.0
        d = float(np.clip(d, -0.5, 0.5))
        offs.append(d * 15.0)
        peak_ratio.append(r0)
    offs = np.array(offs)
    peak_ratio = np.array(peak_ratio)
    print(f'对象数 = {len(offs)}')
    print(f'\n【所选档位相对真实侧视方向的偏航偏差（度）】')
    for q in [5, 25, 50, 75, 95]:
        print(f'  p{q:<3} = {np.percentile(np.abs(offs), q):.2f}')
    print(f'  均值(绝对值) = {np.abs(offs).mean():.2f}   '
          f'|偏差| < 1° 占比 = {100 * (np.abs(offs) < 1).mean():.0f}%   '
          f'>5° 占比 = {100 * (np.abs(offs) > 5).mean():.0f}%')
    print(f'\n【峰值侧视度 max(E0,E1)/E2】中位 = {np.median(peak_ratio):.2f}')
    print(f'  （若某对象峰值只有 ~1，说明它各视角都近端视/形状特殊）')
    low = (peak_ratio < 2.0).sum()
    print(f'  峰值 < 2.0 的对象 = {low}/{len(peak_ratio)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())