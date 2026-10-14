#!/usr/bin/env python3
"""直接看 SS 预测的占据体：是"火车形"还是退化团块？朝向到底在哪根轴？

对同一对象/视角，打印各来源占据体的三轴跨度 + 去顶视图 ASCII，
用于判定"成品竖直板"到底源自 SS 的坐标系，还是源自预测本身已退化。
"""
import os
import sys
import json
import argparse

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers

RC = 'training_data/multiview_1186/renders_cond'
VIEW = 'training_data/multiview_1186/ss_latents_views_mesh_t1'
CANON = 'training_data/multiview_1186/ss_latents/ss_enc_conv3d_16l8_fp16_64'
TARGETS = ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
           'cross_attn.to_kv', 'cross_attn.to_out']
P_INFER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=[0.6, 1.0], rescale_t=5.0)
CH = ' .:-=+*#%@'


def show(occ, label):
    nz = np.argwhere(occ)
    if len(nz) == 0:
        print(f'  {label}: 空'); return
    ext = nz.max(0) - nz.min(0) + 1
    ax = 'xyz'[int(np.argmax(ext))]
    print(f'  {label}: 体素={occ.sum()} 跨度(x,y,z)={tuple(ext)} 长轴={ax}')
    # 侧视：沿 y 取最大值投影（x-z 平面）
    p = occ.max(axis=1)            # (64,64) 即 x-z
    small = torch.from_numpy(p.astype(np.float32))[None, None]
    small = torch.nn.functional.adaptive_avg_pool2d(small, (20, 60))[0, 0].numpy()
    for r in range(20):
        print('      |' + ''.join(CH[min(9, int(v * 10))] for v in small[r]) + '|')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', default=None)
    ap.add_argument('--view', type=int, default=0)
    a = ap.parse_args()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']
    dec = pipe.models['sparse_structure_decoder']
    wdt = next(dec.parameters()).dtype

    def occ_lat(lat):
        dec.cuda()
        return (dec(lat.to(wdt)) > 0)[0, 0].cpu().numpy()

    objs = [o for o in sorted(os.listdir(VIEW))
            if len(os.listdir(os.path.join(VIEW, o))) >= 12
            and os.path.isfile(os.path.join(CANON, o + '.npz'))]
    obj = a.obj or objs[0]
    print(f'=== 对象 {obj} 视角 {a.view} ===')

    print('\n[目标 GT]')
    gv = torch.from_numpy(np.load(os.path.join(VIEW, obj, f'{a.view:03d}.npz'))['mean'])[None].float().cuda()
    gc = torch.from_numpy(np.load(os.path.join(CANON, obj + '.npz'))['z'])[None].float().cuda()
    show(occ_lat(gv), 'GT 视角帧')
    show(occ_lat(gc), 'GT 规范帧')

    # 条件
    from trellis2.datasets import sparse_structure_latent_views as m
    ds = m.ImageConditionedSparseStructureLatentViews(
        json.dumps({'train_data': {'base': './training_data/multiview_1186',
                    'ss_view_latent': './training_data/multiview_1186/ss_latents_views_mesh_t1',
                    'render_cond': './training_data/multiview_1186/renders_cond'}}),
        min_aesthetic_score=4.5, num_views=24, image_size=512, crop_scale=1.0)
    rr, o2, od, valid = ds.instances[objs.index(obj)]
    tf = json.load(open(os.path.join(rr, obj, 'transforms.json')))['frames']
    ten = ds._image(os.path.join(rr, obj, tf[a.view]['file_path']))

    def cond_of(t):
        c = pipe.image_cond_model
        c.image_size = 512
        c.cuda()
        f = c(t[None].cuda())
        return {'cond': f, 'neg_cond': torch.zeros_like(f)}

    def sample(cond, seed=42):
        torch.manual_seed(seed)
        cc = pipe.sample_sparse_structure(cond, 64, 1, dict(P_INFER)).cpu().numpy()
        occ = np.zeros((64, 64, 64), bool)
        for d, h, w in cc[:, 1:]:
            occ[d, h, w] = True
        return occ

    print('\n[预训练模型（无 LoRA）]')
    show(sample(cond_of(ten)), 'base 预测')

    for tag, ck in [
        ('A 视角帧@20k', 'results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts/denoiser_ema0.9999_step0020000.pt'),
        ('B 规范帧@20k', 'results/multiview_1186_ss_flow_lora_64_views_canonical/ckpts/denoiser_ema0.9999_step0020000.pt'),
        ('旧 33k(DINOv2训练)', 'results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0033000.pt'),
    ]:
        if not os.path.isfile(ck):
            print(f'\n[{tag}] 检查点缺失'); continue
        for mm in list(ss.modules()):
            pass
        apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
        sd = torch.load(ck, map_location='cpu', weights_only=True)
        sd.pop('rope_phases', None)
        ss.load_state_dict(sd, strict=False)
        print(f'\n[{tag}]')
        show(sample(cond_of(ten)), f'{tag} 预测')
    return 0


if __name__ == '__main__':
    sys.exit(main())