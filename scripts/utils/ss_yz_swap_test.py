#!/usr/bin/env python3
"""判别：本地 SS 训练目标是否相对"预训练模型/管线约定"整体做了 y/z 交换。

做法：对同一批 (对象, 视角)，取预训练模型的预测占据体 base，比较
  IoU(base,                 GT_view)
  IoU(swap_yz(base),        GT_view)
若后者显著更高，说明本地 GT 与 base/管线约定之间正好差一次 y/z 交换
（即 HANDOFF §6.1-4 提到的那个 blender_dump ↔ dual_grid 的 y/z 对齐，
 但没有被应用到 SS 逐视角体素化脚本上）。

CLI: [--n 8 抽样对象数]
输出：stdout 的 IoU 对照表（原样 vs 交换 y/z，分别对 GT_view / GT_canon）。
安全边界：只读权重与数据，不写任何文件；需 GPU（默认卡 6）。
"""
import os
import sys
import json
import glob
import argparse

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.datasets import sparse_structure_latent_views as m

RC = 'training_data/multiview_1186/renders_cond'
VIEW = 'training_data/multiview_1186/ss_latents_views_mesh_t1'
CANON = 'training_data/multiview_1186/ss_latents/ss_enc_conv3d_16l8_fp16_64'
VOX = 'training_data/multiview_1186/ss_view_voxels_mesh_t1'
P_INFER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=[0.6, 1.0], rescale_t=5.0)


def main():
    """抽样对象：预训练模型预测占据体，分别算"原样/交换 y/z"与本地两种 GT 的 IoU，判断训练目标是否差一次 y/z 交换。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    dec = pipe.models['sparse_structure_decoder']
    wdt = next(dec.parameters()).dtype
    cm = pipe.image_cond_model
    cm.image_size = 512

    def occ_lat(lat):
        """把 GT latent 解码成 64³ 布尔占据体（>0 视为占据）。"""
        dec.cuda()
        return (dec(lat.to(wdt)) > 0)[0, 0].cpu().numpy()

    ds = m.ImageConditionedSparseStructureLatentViews(
        json.dumps({'train_data': {'base': './training_data/multiview_1186',
                    'ss_view_latent': './training_data/multiview_1186/ss_latents_views_mesh_t1',
                    'render_cond': './training_data/multiview_1186/renders_cond'}}),
        min_aesthetic_score=4.5, num_views=24, image_size=512, crop_scale=1.0)

    objs = [o for o in sorted(os.listdir(VIEW))
            if len(glob.glob(os.path.join(VIEW, o, '*.npz'))) >= 12
            and os.path.isfile(os.path.join(CANON, o + '.npz'))]
    picks = [objs[i] for i in np.linspace(0, len(objs) - 1, a.n).astype(int)]

    def sample(cond, seed=42):
        """固定种子采样 SS 并反填成 64³ 布尔占据体。"""
        torch.manual_seed(seed)
        cc = pipe.sample_sparse_structure(cond, 64, 1, dict(P_INFER)).cpu().numpy()
        occ = np.zeros((64, 64, 64), bool)
        for d, h, w in cc[:, 1:]:
            occ[d, h, w] = True
        return occ

    def iou(x, y):
        """体素 IoU；并集为空时返回 0（用 max(1,...) 兜底）。"""
        i = (x & y).sum()
        return i / max(1, (x | y).sum())

    rows = []
    for obj in picks:
        rr, o2, od, valid = ds.instances[objs.index(obj)]
        tf = json.load(open(os.path.join(rr, obj, 'transforms.json')))['frames']
        ten = ds._image(os.path.join(rr, obj, tf[0]['file_path']))
        cm.cuda()
        f = cm(ten[None].cuda())
        base = sample({'cond': f, 'neg_cond': torch.zeros_like(f)})
        gv = occ_lat(torch.from_numpy(
            np.load(os.path.join(VIEW, obj, '000.npz'))['mean'])[None].float().cuda())
        gc = occ_lat(torch.from_numpy(
            np.load(os.path.join(CANON, obj + '.npz'))['z'])[None].float().cuda())
        b_sw = np.swapaxes(base, 1, 2)          # y<->z
        rows.append((obj, iou(base, gv), iou(b_sw, gv),
                     iou(base, gc), iou(b_sw, gc)))
        print(f'  {obj[:12]} ok', flush=True)

    print()
    print('=' * 88)
    print('预训练模型预测 vs 本地 SS 目标：y/z 交换前后的 IoU')
    print('=' * 88)
    print(f'{"":<16}{"vs GT_view 原样":>17}{"vs GT_view 交换yz":>19}'
          f'{"vs GT_canon 原样":>18}{"vs GT_canon 交换yz":>19}')
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f'{"均值":<16}{arr[:,0].mean():>17.3f}{arr[:,1].mean():>19.3f}'
          f'{arr[:,2].mean():>18.3f}{arr[:,3].mean():>19.3f}')
    print(f'{"中位":<16}{np.median(arr[:,0]):>17.3f}{np.median(arr[:,1]):>19.3f}'
          f'{np.median(arr[:,2]):>18.3f}{np.median(arr[:,3]):>19.3f}')
    print()
    print('判读：若"交换 yz"一列显著高于"原样"，则本地 SS 目标相对管线约定整整差一次 y/z 交换。')
    return 0


if __name__ == '__main__':
    sys.exit(main())