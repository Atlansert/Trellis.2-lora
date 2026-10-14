#!/usr/bin/env python3
"""SS 条件一致性 A/B：验证重训后的 LoRA 在"训练条件"下是否真的工作。

对同一批 (对象, 视角) 样本，用同一噪声种子、同一线上采样器，比较：

  cond=DINOv3@512 : 训练与推理实际使用的条件（新 run 的配对）
  cond=DINOv2@518 : 旧 run 训练时使用的条件（现已弃用）

指标：输出 occupancy 与 per-view 真值 latent 的 IoU / 漏检 / 多余。

用法: ss_cond_ab.py --ckpt <SS LoRA .pt> [--loRA none] [--n 6] [--tag 名字]
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
import torch.nn.functional as F
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers
from trellis2.trainers.flow_matching.mixins.image_conditioned import (
    DinoV2LocalFeatureExtractor)

RC = 'training_data/multiview_1186/renders_cond'
SSL = 'training_data/multiview_1186/ss_latents_views_mesh_t1'
TARGETS = ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
           'cross_attn.to_kv', 'cross_attn.to_out']
P_INFER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=[0.6, 1.0], rescale_t=5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='SS LoRA 权重；省略 = 只用预训练')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--view', type=int, default=0)
    a = ap.parse_args()

    print(f'=== {a.tag} | ckpt={a.ckpt} ===', flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']
    dec = pipe.models['sparse_structure_decoder']
    wdt = next(dec.parameters()).dtype

    if a.ckpt:
        n_rep = apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
        sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
        sd.pop('rope_phases', None)
        mis, unexp = ss.load_state_dict(sd, strict=False)
        print(f'  LoRA 注入 {len(n_rep)} 层, missing={len(mis)} unexpected={len(unexp)}',
              flush=True)
        del sd

    dv2 = DinoV2LocalFeatureExtractor(
        model_name='/data/lzx/ReconViaGen/checkpoints_dinov2', image_size=518)
    dv2.cuda()

    def occ_of(lat):
        dec.cuda()
        return (dec(lat.to(wdt)) > 0)[0, 0].cpu().numpy()

    def dataset_image(path, crop_scale=1.0, size=512):
        """复刻新 run 的数据集条件生成（crop_scale=1.0, size=512）"""
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

    def cond_v3(ten):
        m = pipe.image_cond_model
        m.image_size = 512
        m.cuda()
        f = m(ten[None].cuda())
        return {'cond': f, 'neg_cond': torch.zeros_like(f)}

    def cond_v2(ten):
        f = dv2(ten[None].cuda())
        return {'cond': f, 'neg_cond': torch.zeros_like(f)}

    def sample(cond, seed=42):
        torch.manual_seed(seed)
        c = pipe.sample_sparse_structure(cond, 64, 1, dict(P_INFER)).cpu().numpy()
        return set(map(tuple, c[:, 1:]))

    objs = [o for o in sorted(os.listdir(SSL))
            if len(glob.glob(os.path.join(SSL, o, '*.npz'))) >= 12
            and os.path.isfile(os.path.join(RC, o, 'transforms.json'))]
    picks = [(objs[i], a.view) for i in
             np.linspace(0, len(objs) - 1, a.n).astype(int)]
    print(f'  样本: {[p[0][:12] for p in picks]}', flush=True)

    res = {'DINOv3@512 (训练+推理口径)': [], 'DINOv2@518 (旧口径)': []}
    for obj, view in picks:
        tf = json.load(open(os.path.join(RC, obj, 'transforms.json')))['frames']
        p = os.path.join(RC, obj, tf[view]['file_path'])
        gt = torch.from_numpy(
            np.load(os.path.join(SSL, obj, f'{view:03d}.npz'))['mean'])[None].float().cuda()
        gtc = set(map(tuple, np.argwhere(occ_of(gt))))
        ten = dataset_image(p, crop_scale=1.0, size=512)
        for name, cond in [('DINOv3@512 (训练+推理口径)', cond_v3(ten)),
                           ('DINOv2@518 (旧口径)', cond_v2(ten))]:
            cs = sample(cond)
            inter = len(cs & gtc)
            res[name].append((len(cs), len(gtc),
                              inter / max(1, len(cs | gtc)),
                              1 - inter / max(1, len(gtc)),
                              1 - inter / max(1, len(cs))))
        print(f'  {obj[:12]} ok', flush=True)

    print()
    print('=' * 84)
    print(f'{a.tag}: SS occupancy vs 真值（{len(picks)} 样本，64³）')
    print('=' * 84)
    print(f'{"条件":<30}{"预测":>9}{"真值":>9}{"IoU":>9}{"漏检":>9}{"多余":>9}')
    out = {}
    for name, v in res.items():
        iou = float(np.mean([x[2] for x in v]))
        out[name] = iou
        print(f'{name:<30}'
              f'{np.mean([x[0] for x in v]):>9.0f}{np.mean([x[1] for x in v]):>9.0f}'
              f'{iou:>9.3f}{100*np.mean([x[3] for x in v]):>8.1f}%'
              f'{100*np.mean([x[4] for x in v]):>8.1f}%')
    print()
    with open(f'/tmp/ss_cond_ab_{a.tag}.json', 'w') as f:
        json.dump({'tag': a.tag, 'ckpt': a.ckpt, 'iou': out}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())