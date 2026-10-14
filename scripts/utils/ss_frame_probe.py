#!/usr/bin/env python3
"""判定 SS 模型输出的是"规范坐标系"还是"视角坐标系"占用。

对比同一批样本上，模型输出（64³ occupancy）与：
  GT_view  : ss_latents_views_mesh_t1/<obj>/000.npz  —— 视角帧（网格绕 z 旋转 yaw-270 后体素化）
  GT_canon : ss_latents/ss_enc_conv3d_16l8_fp16_64/<obj>.npz —— 规范帧（未旋转）

若显著更接近 GT_view，说明模型预测的是视角帧 —— 推理管线把它当规范帧使用，
输出就会整体转 90°，这解释了 DINOv2 条件下出现"竖直板"的现象。

CLI: [--ckpt LoRA权重] [--cond dinov3|dinov2] [--n 6] [--tag frame] [--gt_view_dir/--gt_canon_dir 覆盖GT目录]
输出：stdout 的 IoU 对照表（vs 视角帧 / vs 规范帧）。
安全边界：只读权重与数据，不写任何文件；需 GPU（默认卡 5）。
"""
import os
import sys
import json
import glob
import argparse

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers
from trellis2.trainers.flow_matching.mixins.image_conditioned import (
    DinoV2LocalFeatureExtractor)

RC = 'training_data/multiview_1186/renders_cond'
VIEW = 'training_data/multiview_1186/ss_latents_views_mesh_t1'
CANON = 'training_data/multiview_1186/ss_latents/ss_enc_conv3d_16l8_fp16_64'
TARGETS = ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
           'cross_attn.to_kv', 'cross_attn.to_out']
P_INFER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=[0.6, 1.0], rescale_t=5.0)


def main():
    """对若干样本：用指定条件编码器采样 SS，分别与"视角帧 GT"和"规范帧 GT"的占据体比 IoU，判断模型学的是哪种帧。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--cond', default='dinov3', choices=['dinov3', 'dinov2'])
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--tag', default='frame')
    ap.add_argument('--gt_view_dir', default=None,
                    help='覆盖"视角帧"GT 目录（用于对齐版目标）')
    ap.add_argument('--gt_canon_dir', default=None,
                    help='覆盖"规范帧"GT 目录')
    a = ap.parse_args()

    global VIEW, CANON
    if a.gt_view_dir:
        VIEW = a.gt_view_dir
    if a.gt_canon_dir:
        CANON = a.gt_canon_dir
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']
    dec = pipe.models['sparse_structure_decoder']
    wdt = next(dec.parameters()).dtype
    if a.ckpt:
        apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
        sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
        sd.pop('rope_phases', None)
        ss.load_state_dict(sd, strict=False)
        del sd
    print(f'=== {a.tag} | cond={a.cond} | ckpt={a.ckpt} ===', flush=True)

    dv2 = None
    if a.cond == 'dinov2':
        dv2 = DinoV2LocalFeatureExtractor(
            model_name='/data/lzx/ReconViaGen/checkpoints_dinov2', image_size=518)
        dv2.cuda()

    def occ_lat(lat):
        """把 GT latent 解码成 64³ 布尔占据体（>0 视为占据）。"""
        dec.cuda()
        return (dec(lat.to(wdt)) > 0)[0, 0].cpu().numpy()

    def dataset_image(path, crop_scale=1.2, size=518):
        """按 alpha 前景 bbox 裁切、缩放并乘 alpha 压黑背景（复刻数据集条件图口径）。"""
        im = Image.open(path).convert('RGBA')
        al = np.array(im.getchannel(3)); ys, xs = np.nonzero(al)
        if len(xs):
            cx = (xs.min() + xs.max()) / 2; cy = (ys.min() + ys.max()) / 2
            hs = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * crop_scale
            im = im.crop((int(cx - hs), int(cy - hs), int(cx + hs), int(cy + hs)))
        im = im.resize((size, size), Image.Resampling.LANCZOS)
        al = torch.from_numpy(np.array(im.getchannel(3))).float() / 255
        x = torch.from_numpy(np.array(im.convert('RGB'))).permute(2, 0, 1).float() / 255
        return x * al.unsqueeze(0)

    def cond_of(ten):
        """按 --cond 选择条件编码器（dinov2 为旧训练口径，否则用管线自带 DINOv3）得到条件特征。"""
        if a.cond == 'dinov2':
            f = dv2(ten[None].cuda())
        else:
            m = pipe.image_cond_model
            m.image_size = 512
            m.cuda()
            f = m(ten[None].cuda())
        return {'cond': f, 'neg_cond': torch.zeros_like(f)}

    def sample(cond, seed=42):
        """固定种子采样 SS，返回占据体坐标集合（便于与 GT 求交并）。"""
        torch.manual_seed(seed)
        c = pipe.sample_sparse_structure(cond, 64, 1, dict(P_INFER)).cpu().numpy()
        return set(map(tuple, c[:, 1:]))

    # 样本挑选：视角帧 npz ≥12 个、且规范帧 npz 存在的对象，再等间隔取 --n 个。
    objs = [o for o in sorted(os.listdir(VIEW))
            if len(glob.glob(os.path.join(VIEW, o, '*.npz'))) >= 12
            and os.path.isfile(os.path.join(CANON, o + '.npz'))]
    picks = [objs[i] for i in np.linspace(0, len(objs) - 1, a.n).astype(int)]
    crop = 1.2 if a.cond == 'dinov2' else 1.0
    size = 518 if a.cond == 'dinov2' else 512

    res = {'vs GT_view（视角帧）': [], 'vs GT_canon（规范帧）': []}
    for obj in picks:
        tf = json.load(open(os.path.join(RC, obj, 'transforms.json')))['frames']
        p = os.path.join(RC, obj, tf[0]['file_path'])
        gv = torch.from_numpy(np.load(os.path.join(VIEW, obj, '000.npz'))['mean'])[None].float().cuda()
        gc = torch.from_numpy(np.load(os.path.join(CANON, obj + '.npz'))['z'])[None].float().cuda()
        sv = set(map(tuple, np.argwhere(occ_lat(gv))))
        sc = set(map(tuple, np.argwhere(occ_lat(gc))))
        cs = sample(cond_of(dataset_image(p, crop, size)))
        for name, gt in [('vs GT_view（视角帧）', sv), ('vs GT_canon（规范帧）', sc)]:
            inter = len(cs & gt)
            res[name].append((inter / max(1, len(cs | gt)), len(gt), len(cs)))
        print(f'  {obj[:12]} ok', flush=True)

    print()
    print('=' * 74)
    print(f'{a.tag} (cond={a.cond})  帧归属判定（{len(picks)} 样本）')
    print('=' * 74)
    print(f'{"对比对象":<26}{"IoU":>9}{"GT体素":>10}{"预测体素":>10}')
    for k, v in res.items():
        print(f'{k:<26}{np.mean([x[0] for x in v]):>9.3f}'
              f'{np.mean([x[1] for x in v]):>10.0f}{np.mean([x[2] for x in v]):>10.0f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())