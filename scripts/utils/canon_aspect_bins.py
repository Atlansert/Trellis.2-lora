#!/usr/bin/env python3
"""检查：训练数据在"002/003 那种宽高比"区间里的目标长轴是否一致。

动机：残余翻转出现在 002（前景宽高比 3.22）与 003（3.92）。需要确认训练数据在
这个区间里给出的是**无歧义**的目标帧（长轴 = index1），否则翻转可以归因于数据歧义。

不必读渲染图：体素侧视度 r = max(E0,E1)/E2 与渲染剪影宽高比高度相关（实测 0.971），
可直接作为宽高比的代理。统计用**置换后**（管线约定）的轴。

运行：无参数（ROOT/VOX 为文件内常量，按当前工作目录解析，需在项目根下执行）。
安全边界：只读 ss_view_voxels_canon_v4/*.npz，不写文件；判据：某区间 index1 占比 >95% 视为无歧义定帧。
"""
import glob
import os
import sys
from collections import Counter

import numpy as np

ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon_v4'
PERM = (2, 0, 1)          # 编码时施加的置换：new[i] = old[PERM[i]]
BINS = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 1e9)]


def main():
    """按侧视度 r 分箱统计"置换后最长轴"的分布，判断该宽高比区间能否无歧义定帧；返回 0。"""
    objs = sorted(os.listdir(os.path.join(ROOT, VOX)))
    table = {b: Counter() for b in BINS}
    n = 0
    for o in objs:
        for f in sorted(glob.glob(os.path.join(ROOT, VOX, o, '*.npz'))):
            c = np.load(f)['coords'].astype(np.int64)
            if not len(c):
                continue
            raw = c.max(0) - c.min(0) + 1
            # 侧视度在原始轴下计算：max(E0,E1)/E2（E2=竖直），作为渲染剪影宽高比的代理。
            r = max(raw[0], raw[1]) / max(raw[2], 1)
            c2 = c[:, PERM]          # 置换到管线约定后再统计最长轴
            e = c2.max(0) - c2.min(0) + 1
            ax = 'xyz'[int(np.argmax(e))]
            for b in BINS:
                if b[0] <= r < b[1]:
                    table[b][ax] += 1
                    break
            n += 1
        if n and len(objs) > 0 and (n % 6000 == 0):
            print(f'  已处理 {n} 样本', flush=True)

    print(f'\n总样本 = {n}（置换后轴约定：index0=竖直、index1=长度、index2=视深）')
    print(f'\n{"侧视度 r 区间":<16}{"n":>7}{"index1 占比":>13}{"其他轴占比":>12}   能否无歧义定帧')
    print('-' * 74)
    for b in BINS:
        c = table[b]
        tot = sum(c.values())
        if tot == 0:
            continue
        p1 = 100 * c['y'] / tot
        other = {k: v for k, v in c.items() if k != 'y'}
        verdict = '✔ 一致' if p1 > 95 else ('~ 有歧义' if p1 > 70 else '✘ 歧义严重')
        print(f'[{b[0]:.1f},{b[1]:.1f}){"":<7}{tot:>7}{p1:>12.1f}%{100 - p1:>11.1f}%   {verdict}   {dict(other)}')
    print('\n参考：data_test 前景宽高比 001=4.19  002=3.22  003=3.92  004=5.68')
    return 0


if __name__ == '__main__':
    sys.exit(main())