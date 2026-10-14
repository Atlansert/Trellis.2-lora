#!/usr/bin/env python3
"""Fast view-aligned SS supervision from the 1024-VAE sparse occupancy grid.

中文说明：
- 用途：从 1024 分辨率 shape latent 的稀疏占据坐标，快速生成逐视角对齐的 SS 监督体素。
- 输入：<root>/shape_latents/<source_name>/<obj>.npz 的 coords + <root>/renders_cond/<obj>/transforms.json 的 yaw。
- 输出：<root>/<output_name>/<obj>/NNN.npz，键 'coords'（uint8，64^3 网格）。
- 关键约定：1024 网格坐标 → 64^3 网格坐标再按视角旋转（v @ rz(yaw-270)），旋转在坐标层完成，
  不重新做网格体素化；已存在的结果在未加 --overwrite 时跳过。
- 用法：python generate_ss_view_from_shape_coords.py --root <dir> [--source_name ...] [--output_name ...]
"""
import argparse, json, os, glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 绕 Z 轴的 3x3 float32 旋转矩阵（deg 为度）。
def rz(deg):
 a=np.deg2rad(deg); c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32)
# 单对象任务：把 1024 网格坐标换算成 64^3 坐标，逐视角旋转后存成适应 SS 的体素坐标。
# 已存在且未 overwrite 的视角跳过；返回 (对象名, 完成视角数)。
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
# CLI 入口：--source_name 指定 shape latent 目录名，--output_name 指定输出目录名，多线程并行。
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--source_name',default='shape_enc_next_dc_f16c32_fp16_1024'); ap.add_argument('--output_name',default='ss_view_voxels_v2'); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args(); root=os.path.abspath(a.root)
 objs=[os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root,'shape_latents',a.source_name,'*.npz')) if os.path.isdir(os.path.join(root,'renders_cond',os.path.basename(p)[:-4]))]; os.makedirs(os.path.join(root,a.output_name),exist_ok=True); print('objects',len(objs),flush=True)
 with ThreadPoolExecutor(a.workers) as ex:
  for n,(o,d) in enumerate(ex.map(one,[(root,x,a.source_name,a.output_name,a.overwrite) for x in objs]),1):
   if n%50==0 or n==len(objs): print(n,'/',len(objs),flush=True)
if __name__=='__main__': main()
