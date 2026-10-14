#!/usr/bin/env python3
"""构建**修正后的规范帧（canonical）latent**：每对象 1 个，长度轴恒为 index1。

动机（实测）：原厂 base 在端视输入上也输出长度轴 = index1，说明预训练模型的约定是
**规范帧**而非相机帧。旧的 canonical run（run B）失败，是因为它用的规范帧 latent 来自
**错误的体素化**（长度轴随资产类别 71/29 分裂），目标仍不可统一。

做法：从已修正相机约定的逐视角 latent `ss_latents_views_canon` 中，
**取"最侧视"那一帧**作为该对象的规范帧目标——在侧视帧上，物体的长度轴天然落在
画面水平轴（置换后 = index1），因此所有对象共享同一轴约定。

输出：<root>/<output_name>/<obj>.npz，键 'mean'（与数据集 canonical 分支一致）。
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image


def aspect(path):
    a = np.array(Image.open(path).convert('RGBA').getchannel('A').resize((128, 128), Image.BOX))
    ys, xs = np.nonzero(a > 32)
    if len(xs) < 8:
        return 0.0
    return (xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1)


def one(job):
    root, obj, src, dst = job
    rp = sorted(glob.glob(os.path.join(root, 'renders_cond_t', obj, '*.png')))
    if not rp:
        return obj, -1, 'no renders'
    ars = [aspect(p) for p in rp]
    v = int(np.argmax(ars))
    sp = os.path.join(root, src, obj, f'{v:03d}.npz')
    if not os.path.isfile(sp):
        # 退回到任一存在的视角
        cand = sorted(glob.glob(os.path.join(root, src, obj, '*.npz')))
        if not cand:
            return obj, -1, 'no latent'
        sp = cand[0]
        v = int(os.path.basename(sp)[:3])
    z = np.load(sp)
    key = 'mean' if 'mean' in z.files else 'z'
    np.savez_compressed(os.path.join(dst, obj + '.npz'), mean=z[key])
    return obj, v, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='training_data/multiview_1186')
    ap.add_argument('--src', default='ss_latents_views_canon')
    ap.add_argument('--output_name', default='ss_latents_canon')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    dst = os.path.join(root, a.output_name)
    os.makedirs(dst, exist_ok=True)
    objs = [o for o in sorted(os.listdir(os.path.join(root, a.src)))
            if os.path.isdir(os.path.join(root, a.src, o))]
    print(f'objects={len(objs)} -> {dst}', flush=True)
    jobs = [(root, o, a.src, dst) for o in objs]
    ok = 0
    vs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (o, v, e) in enumerate(ex.map(one, jobs), 1):
            if e:
                print(f'{o}: {e}', flush=True)
            else:
                ok += 1
                vs.append(v)
            if n % 200 == 0 or n == len(jobs):
                print(f'{n}/{len(jobs)} ok={ok}', flush=True)
    vs = np.array(vs)
    print(f'完成 {ok}/{len(objs)}；所选视角分布（0..23）: '
          f'{np.bincount(vs, minlength=24).tolist()}', flush=True)


if __name__ == '__main__':
    main()