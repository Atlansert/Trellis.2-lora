#!/usr/bin/env python3
"""Encode per-view 64^3 occupancy grids with the local TRELLIS.2 SS VAE.

中文说明：
- 用途：把逐视角体素网格（<root>/<voxel_name>/<obj>/NNN.npz 的 coords）编码成 SS latent。
  这是旧轴约定版本（未做 canon 修正），正式的规范帧路径见 encode_ss_view_latents_canon.py。
- 输入：--root 数据根目录、--voxel_name 体素目录名、--encoder SS 编码器路径（需要 GPU）。
- 输出：<root>/<output_name>/<obj>/NNN.npz，键 'mean'（沿用 TRELLIS.1 的键名）。
- 关键约定：不做轴置换；坐标裁剪到 [0,res-1]；编码取确定性均值（sample_posterior=False）。
- 用法：python encode_ss_view_latents.py --root <dir> [--voxel_name ...] [--output_name ...]
"""
import argparse, os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm
import trellis2.models as models

# CLI 入口：逐批把体素坐标栅格化为 res^3 二值网格 → SS 编码器前向 → 存 'mean'。
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--voxel_name', default='ss_view_voxels_v2')
    ap.add_argument('--output_name', default='ss_latents_views_v2')
    ap.add_argument('--encoder', default='/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--resolution', type=int, default=64)
    args=ap.parse_args()
    root=os.path.abspath(args.root)
    inp=os.path.join(root,args.voxel_name); out=os.path.join(root,args.output_name)
    os.makedirs(out,exist_ok=True)
    enc=models.from_pretrained(args.encoder).eval().cuda()
    files=sorted(glob.glob(os.path.join(inp,'*','*.npz')))
    print(f'files={len(files)} encoder={args.encoder}',flush=True)
    for st in tqdm(range(0,len(files),args.batch_size), desc='Encoding view latents'):
        batch_files=files[st:st+args.batch_size]; grids=[]
        for p in batch_files:
            c=np.load(p)['coords'].astype(np.int64)
            # Open3D's inclusive max-bound can emit index=res at the upper
            # boundary; TRELLIS.1's voxel grid is strictly [0,res-1].
            c=np.clip(c, 0, args.resolution-1)
            x=torch.zeros(args.resolution,args.resolution,args.resolution,dtype=torch.float32)
            if len(c): x[c[:,0],c[:,1],c[:,2]]=1
            grids.append(x)
        x=torch.stack(grids,0)[:,None].cuda()
        z=enc(x,sample_posterior=False).cpu().numpy()
        for p,zz in zip(batch_files,z):
            obj=os.path.basename(os.path.dirname(p)); os.makedirs(os.path.join(out,obj),exist_ok=True)
            # TRELLIS.1 stores the encoder output under ``mean``.
            np.savez_compressed(os.path.join(out,obj,os.path.basename(p)), mean=zz)
    print('done',flush=True)
if __name__=='__main__': main()
