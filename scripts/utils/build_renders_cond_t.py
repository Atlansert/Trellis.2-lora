#!/usr/bin/env python3
"""构建 renders_cond_t：633 个不透明对象的条件图补上 alpha（透明背景），
549 个本来就是透明的对象用符号链接直接复用。

为什么不是"重渲"：data_for_train 里 327 个 .obj 的 .mtl 全部缺失（0/327），
重渲会得到无贴图的灰模，比存档更差。存档渲染已带正确贴图/相机/色调，只差 alpha。
本脚本精确补上 alpha，RGB 原样保留（训练/推理侧 x*alpha 会把背景变黑）。

输出结构（与 renders_cond 同构，可直接替换数据集的 render_cond）：
    renders_cond_t/<sha>/000_cam_000.png ... 023_cam_345.png  + real_*.png + transforms.json
"""
import os
import sys
import csv
import json
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = 'training_data/multiview_1186'
TOL = 3.0


def border_bg(arr):
    h, w = arr.shape[:2]
    k = max(2, min(h, w) // 64)
    ring = np.concatenate([arr[:k].reshape(-1, 3), arr[-k:].reshape(-1, 3),
                           arr[:, :k].reshape(-1, 3), arr[:, -k:].reshape(-1, 3)])
    return np.median(ring, axis=0)


def derive_mask(rgb):
    bg = border_bg(rgb)
    m = np.abs(rgb - bg).max(axis=2) > TOL
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)), iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(m)


def do_object(job):
    sha, outroot = job
    src = os.path.join(ROOT, 'renders_cond', sha)
    dst = os.path.join(outroot, sha)
    os.makedirs(dst, exist_ok=True)
    tf = json.load(open(os.path.join(src, 'transforms.json')))
    n_render = n_real = n_hole = 0
    for f in tf['frames']:
        sp = os.path.join(src, f['file_path'])
        if not os.path.isfile(sp):
            continue
        dp = os.path.join(dst, f['file_path'])
        if 'yaw_deg' in f:
            arr = np.array(Image.open(sp).convert('RGB')).astype(np.float32)
            m = derive_mask(arr)
            n_hole += int(m.sum())
            out = np.dstack([arr, m.astype(np.float32) * 255])
            Image.fromarray(out.astype(np.uint8), 'RGBA').save(dp)
            n_render += 1
        else:
            shutil.copy2(sp, dp)
            n_real += 1
    with open(os.path.join(dst, 'transforms.json'), 'w') as fh:
        json.dump(tf, fh, indent=2)
    return sha, n_render, n_real


def classify():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'metadata.csv'))))
    opaque, transp = [], []
    for r in rows:
        p = os.path.join(ROOT, 'renders_cond', r['sha256'], '000_cam_000.png')
        if not os.path.isfile(p):
            continue
        a = np.array(Image.open(p).convert('RGBA'))[:, :, 3]
        (opaque if (a > 0).mean() >= 0.99 else transp).append(r['sha256'])
    return opaque, transp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='renders_cond_t')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    outroot = os.path.join(ROOT, a.out)
    os.makedirs(outroot, exist_ok=True)
    opaque, transp = classify()
    if a.limit:
        opaque = opaque[:a.limit]
    print(f'不透明 {len(opaque)} 个（补 alpha），透明 {len(transp)} 个（符号链接）', flush=True)

    jobs = [(s, outroot) for s in opaque]
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for sha, nr, nreal in ex.map(do_object, jobs):
            done += 1
            if done % 100 == 0:
                print(f'  {done}/{len(jobs)}', flush=True)
    print(f'补 alpha 完成 {done} 个', flush=True)

    n_link = 0
    for sha in transp:
        dst = os.path.join(outroot, sha)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        os.symlink(os.path.join('..', 'renders_cond', sha), dst)
        n_link += 1
    print(f'透明对象链接 {n_link} 个', flush=True)
    total = len([d for d in os.listdir(outroot)])
    print(f'{outroot} 共 {total} 个对象目录', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())