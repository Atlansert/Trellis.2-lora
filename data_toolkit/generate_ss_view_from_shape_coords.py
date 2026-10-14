#!/usr/bin/env python3
"""Fast view-aligned SS supervision from the 1024-VAE sparse occupancy grid."""
import argparse, json, os, glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def rz(deg):
 a=np.deg2rad(deg); c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32)
def one(job):
 root,obj,src_name,out_name,overwrite=job; src=os.path.join(root,'shape_latents',src_name,obj+'.npz'); od=os.path.join(root,'renders_cond',obj); out=os.path.join(root,out_name,obj); os.makedirs(out,exist_ok=True)
 try: frames=json.load(open(os.path.join(od,'transforms.json')))['frames']; c=np.load(src)['coords'].astype(np.float32)
 except Exception: return obj,0
 # 1024 shape coordinates decode to 64^3 SS grid coordinates.
 p=(c+0.5)/64.0-0.5; done=0
 for i,fr in enumerate(frames[:24]):
  q=os.path.join(out,f'{i:03d}.npz')
  if os.path.exists(q) and not overwrite: done+=1; continue
  yaw=float(fr.get('yaw_deg',i*15)); cc=np.floor(((p@rz(yaw-270))+0.5)*64).astype(np.int32); cc=np.unique(np.clip(cc,0,63),axis=0).astype(np.uint8)
  np.savez_compressed(q,coords=cc); done+=1
 return obj,done
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--source_name',default='shape_enc_next_dc_f16c32_fp16_1024'); ap.add_argument('--output_name',default='ss_view_voxels_v2'); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args(); root=os.path.abspath(a.root)
 objs=[os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root,'shape_latents',a.source_name,'*.npz')) if os.path.isdir(os.path.join(root,'renders_cond',os.path.basename(p)[:-4]))]; os.makedirs(os.path.join(root,a.output_name),exist_ok=True); print('objects',len(objs),flush=True)
 with ThreadPoolExecutor(a.workers) as ex:
  for n,(o,d) in enumerate(ex.map(one,[(root,x,a.source_name,a.output_name,a.overwrite) for x in objs]),1):
   if n%50==0 or n==len(objs): print(n,'/',len(objs),flush=True)
if __name__=='__main__': main()
