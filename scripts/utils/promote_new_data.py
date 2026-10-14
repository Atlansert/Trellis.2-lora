#!/usr/bin/env python3
"""校验暂存区渲染结果，通过后移入 data_for_train。

校验项（每个单元）：
  * mesh_textured_pbr.glb 存在且为有效 glTF
  * transforms.json 存在且 frames == 24，yaw 覆盖 0..345
  * 24 张 00N_cam_XXX.png 存在
  * 渲染图为 RGBA 且具真实 alpha（与官方 render_cond.py / 推理约定一致）
  * 物体填充比在合理区间
不通过的单元不移动，单独列出。

CLI: [--root /data/lzx/TRELLIS.2] [--staged new_data_staged] [--dest data_for_train] [--apply]
输出：stdout 校验报告（通过数、填充比统计、未通过原因）。
安全边界：默认只校验不动文件；只有加 --apply 才把通过单元 shutil.move 进 --dest（目标同名则跳过），
          移动后暂存区不再保留该单元。
"""
import os
import re
import sys
import json
import glob
import shutil
import struct
import argparse

LABELS = [f'cam_{y:03d}' for y in range(0, 360, 15)]


def probe_glb(path):
    """快速校验 GLB 头(magic/声明长度)与首个 JSON 块，要求至少一个 mesh primitive；返回 (是否有效, 原因)。"""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(12)
        if len(head) < 12:
            return False, 'file too small'
        magic, version, length = struct.unpack('<4sII', head)
        if magic != b'glTF' or length != size:
            return False, 'bad glb header'
        clen, ctype = struct.unpack('<I4s', f.read(8))
        if ctype != b'JSON':
            return False, 'first chunk not JSON'
        doc = json.loads(f.read(clen).decode('utf-8'))
    prims = sum(len(m.get('primitives') or []) for m in (doc.get('meshes') or []))
    return (prims > 0), ('ok' if prims else 'no mesh primitives')


def check_unit(d, need_alpha=True):
    """返回 (ok, 原因列表, 统计)。逐项校验：GLB 有效性、transforms.json 的 24 帧与 yaw 序列、
    24 张渲染图是否齐全、alpha 是否为真实透明背景、物体填充比是否在合理区间。
    """
    probs = []
    mesh = os.path.join(d, 'mesh_textured_pbr.glb')
    if not os.path.isfile(mesh):
        probs.append('缺 mesh_textured_pbr.glb')
    else:
        good, why = probe_glb(mesh)
        if not good:
            probs.append(f'glb 无效: {why}')
    tfp = os.path.join(d, 'transforms.json')
    if not os.path.isfile(tfp):
        probs.append('缺 transforms.json')
        return False, probs, {}
    tf = json.load(open(tfp))
    frames = tf.get('frames', [])
    if len(frames) != 24:
        probs.append(f'frames={len(frames)} (期望 24)')
    yaws = [f.get('yaw_deg') for f in frames]
    if yaws != list(range(0, 360, 15)):
        probs.append(f'yaw 序列异常: {yaws[:6]}...')
    missing = [f['file_path'] for f in frames
               if not os.path.isfile(os.path.join(d, f['file_path']))]
    if missing:
        probs.append(f'缺 {len(missing)} 张渲染图')
    stat = {}
    if not missing and need_alpha:
        import numpy as np
        from PIL import Image
        fills, covs = [], []
        for f in frames[::6]:
            im = Image.open(os.path.join(d, f['file_path'])).convert('RGBA')
            a = np.array(im)[:, :, 3]
            cov = float((a > 0).mean())
            nz = np.nonzero(a)
            if len(nz[0]) == 0:
                fills.append(0.0); covs.append(cov); continue
            w = nz[1].max() - nz[1].min(); h = nz[0].max() - nz[0].min()
            fills.append(max(w, h) / max(im.size))
            covs.append(cov)
        stat = {'alpha 覆盖率': sum(covs) / len(covs),
                '物体填充比': sum(fills) / len(fills)}
        if stat['alpha 覆盖率'] > 0.95:
            probs.append(f"alpha 覆盖率 {stat['alpha 覆盖率']:.3f} —— 非透明背景")
        if not (0.30 <= stat['物体填充比'] <= 1.0):
            probs.append(f"物体填充比 {stat['物体填充比']:.3f} 异常")
    return (not probs), probs, stat


def main():
    """校验暂存区全部单元；未通过的不动，通过的在 --apply 时移入 data_for_train。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/lzx/TRELLIS.2')
    ap.add_argument('--staged', default='new_data_staged')
    ap.add_argument('--dest', default='data_for_train')
    ap.add_argument('--apply', action='store_true', help='真正移动（否则只校验）')
    a = ap.parse_args()
    root = a.root
    staged = os.path.join(root, a.staged)
    dest = os.path.join(root, a.dest)
    units = sorted(d for d in os.listdir(staged)
                   if os.path.isdir(os.path.join(staged, d))
                   and not d.startswith('_'))
    ok, bad = [], []
    fills = []
    for u in units:
        good, probs, stat = check_unit(os.path.join(staged, u))
        if good:
            ok.append(u)
            if '物体填充比' in stat:
                fills.append(stat['物体填充比'])
        else:
            bad.append((u, probs))
    print(f'校验通过 {len(ok)} / {len(units)}')
    if fills:
        import statistics
        print(f'物体填充比: 中位 {statistics.median(fills):.3f} '
              f'范围 {min(fills):.3f}-{max(fills):.3f}')
    if bad:
        print(f'\n未通过 {len(bad)} 个（不移动）:')
        for u, p in bad:
            print(f'  ✗ {u}: {"; ".join(p)}')
    if not a.apply:
        print('\n（仅校验模式，未移动。加 --apply 执行移动）')
        return
    os.makedirs(dest, exist_ok=True)
    moved = 0
    for u in ok:
        src, dst = os.path.join(staged, u), os.path.join(dest, u)
        if os.path.exists(dst):
            print(f'  [跳过] 目标已存在: {u}')
            continue
        shutil.move(src, dst)
        moved += 1
    print(f'\n已移入 {dest}: {moved} 个单元')
    print(f'data_for_train 现有单元数: '
          f'{len([d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))])}')


if __name__ == '__main__':
    main()