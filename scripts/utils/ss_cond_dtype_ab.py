#!/usr/bin/env python3
"""诊断：条件编码精度（训练 bf16 vs 推理 fp32）以及条件生成路径对 SS LoRA 效果的影响。

训练侧 _init_image_cond_model 会把条件模型转成 bf16；推理管线不转（保持 fp32）。
这里固定 LoRA 与噪声种子，只切换条件来源，测 SS occupancy IoU。

CLI: --ckpt 必填(LoRA/SS checkpoint) [--n 6] [--tag dtype]
输出：stdout 对照表 + 写 /tmp/dtype_<tag>.json（只存平均 IoU）。
安全边界：只读权重与数据，除 /tmp 下该 json 外不写文件；需 GPU（默认卡 5）。
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

RC = 'training_data/multiview_1186/renders_cond'
SSL = 'training_data/multiview_1186/ss_latents_views_mesh_t1'
TARGETS = ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
           'cross_attn.to_kv', 'cross_attn.to_out']
P_INFER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=[0.6, 1.0], rescale_t=5.0)


def main():
    """对若干样本，分别用 fp32(推理口径) 与 bf16(训练口径) 生成条件再采样 SS，与 GT 占据体比 IoU 并对比。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--tag', default='dtype')
    a = ap.parse_args()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']
    dec = pipe.models['sparse_structure_decoder']
    wdt = next(dec.parameters()).dtype      # 跟随解码器实际 dtype，避免精度不匹配

    apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    sd.pop('rope_phases', None)
    ss.load_state_dict(sd, strict=False)
    del sd
    print(f'=== {a.tag} | {a.ckpt} ===', flush=True)

    cm = pipe.image_cond_model
    cm.image_size = 512

    def occ_of(lat):
        """把 latent 解码成 64³ 布尔占据体（>0 视为占据）。"""
        dec.cuda()
        return (dec(lat.to(wdt)) > 0)[0, 0].cpu().numpy()

    def dataset_image(path, crop_scale=1.0, size=512):
        """按 alpha 前景 bbox 裁切、缩放，并乘 alpha 把背景压黑（复刻数据集的条件图口径）。"""
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

    def cond_of(ten, dtype):
        """用指定精度跑条件编码器，返回 {'cond': 特征, 'neg_cond': 全 0}。"""
        cm.model.to(device=torch.cuda.current_device())
        cm.model.to(dtype=dtype)
        f = cm(ten[None].cuda())
        return {'cond': f.float(), 'neg_cond': torch.zeros_like(f.float())}

    def sample(cond, seed=42):
        """固定种子采样 SS，返回占据体坐标集合（便于与 GT 求交并）。"""
        torch.manual_seed(seed)
        c = pipe.sample_sparse_structure(cond, 64, 1, dict(P_INFER)).cpu().numpy()
        return set(map(tuple, c[:, 1:]))

    objs = [o for o in sorted(os.listdir(SSL))
            if len(glob.glob(os.path.join(SSL, o, '*.npz'))) >= 12
            and os.path.isfile(os.path.join(RC, o, 'transforms.json'))]
    picks = [objs[i] for i in np.linspace(0, len(objs) - 1, a.n).astype(int)]
    print(f'  样本 {len(picks)} 个', flush=True)

    res = {}
    for obj in picks:
        tf = json.load(open(os.path.join(RC, obj, 'transforms.json')))['frames']
        p = os.path.join(RC, obj, tf[0]['file_path'])
        gt = torch.from_numpy(np.load(os.path.join(SSL, obj, '000.npz'))['mean'])[None].float().cuda()
        gtc = set(map(tuple, np.argwhere(occ_of(gt))))
        ten = dataset_image(p, crop_scale=1.0, size=512)
        for name, dt in [('条件 fp32（推理口径）', torch.float32),
                         ('条件 bf16（训练口径）', torch.bfloat16)]:
            cs = sample(cond_of(ten, dt))
            inter = len(cs & gtc)
            res.setdefault(name, []).append(
                (len(cs), len(gtc), inter / max(1, len(cs | gtc))))
        print(f'  {obj[:12]} ok', flush=True)

    print()
    print('=' * 70)
    print(f'{a.tag}: 条件精度的影响（{len(picks)} 样本）')
    print('=' * 70)
    print(f'{"条件":<26}{"预测":>9}{"真值":>9}{"IoU":>9}')
    for name, v in res.items():
        print(f'{name:<26}{np.mean([x[0] for x in v]):>9.0f}'
              f'{np.mean([x[1] for x in v]):>9.0f}{np.mean([x[2] for x in v]):>9.3f}')
    json.dump({k: float(np.mean([x[2] for x in v])) for k, v in res.items()},
              open(f'/tmp/dtype_{a.tag}.json', 'w'), indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())