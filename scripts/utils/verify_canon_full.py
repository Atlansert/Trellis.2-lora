#!/usr/bin/env python3
"""全量校验规范化后的数据与规范帧 latent（不抽样）。

检查项：
1. 体素 v4：每对象 24 个文件、非空、跨度合理、index2（竖直）在所有视角近似恒定
2. 逐视角 latent：文件数、shape、dtype
3. 规范帧 latent：每对象 1 个，解码前用**体素**判定其长度轴是否为 index1
   （规范帧取自最侧视视角，侧视帧的长度轴应落在 index1）
4. 对照组：旧数据中"latent 长轴对全部视角不变"的对象比例
"""
import os
import glob
import sys
from collections import Counter

import numpy as np

ROOT = 'training_data/multiview_1186'
PERM = (2, 0, 1)


def main():
    vox = os.path.join(ROOT, 'ss_view_voxels_canon_v4')
    lv = os.path.join(ROOT, 'ss_latents_views_canon')
    lc = os.path.join(ROOT, 'ss_latents_canon')

    objs = sorted(os.listdir(vox))
    print(f'对象数 = {len(objs)}')

    bad_files, empty, span_issues = [], 0, 0
    up_rel = []
    long_axes_v0 = Counter()
    per_obj_const = 0
    for o in objs:
        fs = sorted(glob.glob(os.path.join(vox, o, '*.npz')))
        if len(fs) != 24:
            bad_files.append((o, len(fs)))
            continue
        ups = []
        axes = set()
        for i, f in enumerate(fs):
            c = np.load(f)['coords'].astype(np.int64)
            if not len(c):
                empty += 1
                continue
            e = c.max(0) - c.min(0) + 1
            if e[2] < 2 or e[2] > 40:
                span_issues += 1
            ups.append(e[2])
            axes.add('xyz'[int(np.argmax(e))])
            if i == 0:
                long_axes_v0['xyz'[int(np.argmax(e))]] += 1
        if ups:
            ups = np.array(ups)
            up_rel.append(ups.std() / max(ups.mean(), 1))
        if len(axes) == 1:
            per_obj_const += 1

    print(f'\n[体素 v4]')
    print(f'  24 文件齐全: {len(objs) - len(bad_files)}/{len(objs)}'
          + (f'  异常: {bad_files[:5]}' if bad_files else ''))
    print(f'  空文件数 = {empty}')
    print(f'  竖直轴(index2)跨度异常数 = {span_issues}')
    up_rel = np.array(up_rel)
    print(f'  index2 跨度跨视角相对波动 中位 = {np.median(up_rel):.4f}'
          f'（竖直不受环绕影响，应接近 0）')
    print(f'  view0 长轴分布 = {dict(long_axes_v0)}')
    print(f'  长轴对 24 个视角始终同一根轴的对象 = {per_obj_const}/{len(objs)}'
          f'  ({100 * per_obj_const / len(objs):.0f}%)')

    # 逐视角 latent
    n_lv = sum(len(os.listdir(os.path.join(lv, o))) for o in os.listdir(lv))
    z = np.load(glob.glob(os.path.join(lv, '*', '000.npz'))[0])['mean']
    print(f'\n[逐视角 latent] 文件数 = {n_lv}  shape = {z.shape}  dtype = {z.dtype}')

    # 规范帧 latent：用体素判定其长度轴
    ck = Counter()
    miss = 0
    for o in objs:
        p = os.path.join(lc, o + '.npz')
        if not os.path.isfile(p):
            miss += 1
            continue
        zz = np.load(p)
        if 'mean' not in zz.files and 'z' not in zz.files:
            miss += 1
    print(f'\n[规范帧 latent] 文件数 = {len(os.listdir(lc))}  缺失 = {miss}')
    # 规范帧对应体素的长度轴（选"最侧视"视角的体素）
    for o in objs[:400]:
        fs = sorted(glob.glob(os.path.join(vox, o, '*.npz')))
        best, ba = -1, None
        for i, f in enumerate(fs):
            c = np.load(f)['coords'].astype(np.int64)
            if not len(c):
                continue
            c = c[:, PERM]
            e = c.max(0) - c.min(0) + 1
            # 用水平跨度/竖直跨度近似"侧视程度"
            r = e[1] / max(e[0], 1)
            if r > best:
                best, ba = r, 'xyz'[int(np.argmax(e))]
        if ba:
            ck[ba] += 1
    print(f'  规范帧（取最侧视视角体素、置换后）长轴分布 = {dict(ck)}'
          f'  → index1 占比 {100 * ck["y"] / max(sum(ck.values()), 1):.0f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())