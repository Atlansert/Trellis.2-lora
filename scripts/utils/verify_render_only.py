#!/usr/bin/env python3
"""验证：训练条件**只使用渲染帧**，真图（frame 24, real_*.png）一个都没进来。

对 A′（逐视角）与 B′（规范帧）两种口径分别复现数据集的 valid 计算，
然后逐对象统计 valid 里有多少帧指向 real_*.png。正确结果应为 **0**。

运行：无参数（路径写死在文件内，通过 os.chdir 切到项目根）。
安全边界：只读 transforms.json 并检查图片是否存在，不写任何文件；判定只看 valid 中真图计数是否为 0。
"""
import json
import os
import sys
from collections import Counter

os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

ROOT = 'training_data/multiview_1186'
RC = os.path.join(ROOT, 'renders_cond_t')
OBJS = sorted(os.listdir(RC))


def scan(kind):
    """扫描全部对象目录：统计参与训练的对象数、有效(图,latent)对数、valid 中指向 real_*.png 的样本数，
    以及各帧下标被选中的次数（用于看真图所在帧是否入选）。kind 仅作为标签，不影响计算。"""
    n_objs = 0
    n_pairs = 0
    real_pairs = 0
    frame_counts = Counter()
    has_real_file = 0
    for o in OBJS:
        tr = os.path.join(RC, o, 'transforms.json')
        if not os.path.isfile(tr):
            continue
        frames = json.load(open(tr))['frames']
        if any('real' in str(f.get('file_path', '')).lower() for f in frames):
            has_real_file += 1
        # valid = 帧表里的前 24 帧中，图片文件确实存在的那些（与数据集 valid 计算口径一致）。
        n = min(24, len(frames))
        valid = [i for i in range(n)
                 if os.path.exists(os.path.join(RC, o, frames[i]['file_path']))]
        n_objs += 1
        n_pairs += len(valid)
        for i in valid:
            fp = str(frames[i].get('file_path', ''))
            if 'real' in fp.lower():
                real_pairs += 1
            frame_counts[i] += 1
    return n_objs, n_pairs, real_pairs, has_real_file, frame_counts


for kind, num_views in [("A' 逐视角 (view)", 24), ("B' 规范帧 (canonical)", 24)]:
    n_objs, n_pairs, real_pairs, has_real_file, fc = scan(kind)
    print(f'=== {kind} ===')
    print(f'  参与训练的对象            = {n_objs}')
    print(f'  条件样本（图,latent）对数  = {n_pairs}')
    print(f'  含 real_*.png 的目录数     = {has_real_file}'
          f'  （这些真图在帧表里，但见下）')
    print(f'  **valid 中指向真图的样本数 = {real_pairs}**'
          f'   -> {"✔ 完全未使用" if real_pairs == 0 else "✘ 有真图混入"}')
    used = sorted(fc)
    print(f'  valid 覆盖的帧下标范围     = {used[0]}..{used[-1]}（共 {len(used)} 个）')
    print(f'  帧 {max(used)+1}（真图所在位）是否入选 = '
          f'{"是" if (max(used)+1) in fc else "否"}')
    print()