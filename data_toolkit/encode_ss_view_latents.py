#!/usr/bin/env python3
"""Encode per-view 64^3 occupancy grids with the local TRELLIS.2 SS VAE."""
import argparse, os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm
import trellis2.models as models

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
