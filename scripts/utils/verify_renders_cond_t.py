#!/usr/bin/env python3
"""验证 renders_cond_t：训练侧条件图 vs 推理侧条件图（**同一张图**逐像素比较）。

注意：不能用 dataset[i]（它会随机抽视角），必须直接调 _image(同一个文件)。
"""
import os
import sys
import json
import argparse

os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.datasets import sparse_structure_latent_views as m
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

RC = 'training_data/multiview_1186/renders_cond'
RT = 'training_data/multiview_1186/renders_cond_t'


class Dummy:
    low_vram = False
    _device = 'cpu'


ROOTS = lambda rc: json.dumps({'train_data': {
    'base': './training_data/multiview_1186',
    'ss_view_latent': './training_data/multiview_1186/ss_latents_views_mesh_t1',
    'render_cond': rc}})


def infer_cond(path, size=512):
    img = Image.open(path).convert('RGBA')
    out = Trellis2ImageTo3DPipeline.preprocess_image(Dummy(), img)
    out = out.resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(out).astype(np.float32) / 255).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--view', type=int, default=0)
    a = ap.parse_args()
    ds_old = m.ImageConditionedSparseStructureLatentViews(
        ROOTS('./training_data/multiview_1186/renders_cond'),
        min_aesthetic_score=4.5, num_views=24, image_size=512, crop_scale=1.0)
    ds_new = m.ImageConditionedSparseStructureLatentViews(
        ROOTS('./training_data/multiview_1186/renders_cond_t'),
        min_aesthetic_score=4.5, num_views=24, image_size=512, crop_scale=1.0)

    opaque, transp = [], []
    for rr, obj, od, v in ds_old.instances:
        p = os.path.join(RC, obj, '000_cam_000.png')
        if not os.path.isfile(p):
            continue
        al = np.array(Image.open(p).convert('RGBA'))[:, :, 3]
        (opaque if (al > 0).mean() >= 0.99 else transp).append(obj)
    picks = ([opaque[i] for i in np.linspace(0, len(opaque) - 1, a.n).astype(int)] +
             [transp[i] for i in np.linspace(0, len(transp) - 1, 3).astype(int)])

    print(f'不透明 {len(opaque)} / 透明 {len(transp)}；抽 {len(picks)} 个（视角 {a.view}）')
    print()
    print(f'{"对象":<14}{"类型":>7}{"旧:非零占比":>13}{"新:非零占比":>13}'
          f'{"新:训练vs推理 |Δ|":>20}{"新:相关性":>12}')
    on, nn, dd, cc = [], [], [], []
    for obj in picks:
        typ = '不透明' if obj in opaque else '透明'
        fn = f'{a.view:03d}_cam_{a.view * 15:03d}.png'
        po, pn = os.path.join(RC, obj, fn), os.path.join(RT, obj, fn)
        if not (os.path.isfile(po) and os.path.isfile(pn)):
            continue
        fo = float((ds_old._image(po).abs().sum(0) > 0).float().mean())
        tn = ds_new._image(pn)
        fnn = float((tn.abs().sum(0) > 0).float().mean())
        ti = infer_cond(pn)
        d = float((tn - ti).abs().mean() * 255)
        c = float(np.corrcoef(tn.numpy().ravel(), ti.numpy().ravel())[0, 1])
        on.append(fo); nn.append(fnn); dd.append(d); cc.append(c)
        print(f'{obj[:12]:<14}{typ:>7}{fo:>13.3f}{fnn:>13.3f}{d:>20.2f}{c:>12.3f}')
    print()
    print(f'均值  旧非零占比={np.mean(on):.3f}（1.0=灰白背景没被消掉）  '
          f'新非零占比={np.mean(nn):.3f}  '
          f'新训练vs推理 |Δ|={np.mean(dd):.2f}/255 相关={np.mean(cc):.3f}')
    print()
    print('解读：新口径下"训练时喂的条件图"与"推理时喂的"应几乎一致（|Δ| 小、相关≈1），')
    print('      且背景已被 alpha 消掉（非零占比远小于 1）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())