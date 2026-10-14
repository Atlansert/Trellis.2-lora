"""CPU fixed-axis three-image comparison; approximate triangle painter, not acceptance score."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
r=Path(args.root); out=Path(args.out)
if out.exists(): raise FileExistsError(out)
rows=[('003','axis10_20k_003_remediation_20261013')]
canvas=Image.new('RGB',(1530,320),(22,24,28));draw=ImageDraw.Draw(canvas)
draw.text((10,8),'Input vs old33k SS vs axis10 EMA20k; seed42, shape20k fixed, original GLB coordinates. No axis correction.',fill='white')
draw.text((10,26),'Same geometry length scale; CPU orthographic approximation. Inspect the complete GLB for final geometry acceptance.',fill='white')
for row,(n,tag) in enumerate(rows):
 y=55+row*255
 im=Image.open(r/'data_test'/f'{n}.png').convert('RGB');a=np.asarray(im);ys,xs=np.nonzero(a.max(2)>16)
 ratio=(xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)
 crop=im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1));crop=crop.resize((420,round(420/ratio)),Image.Resampling.LANCZOS)
 tile=Image.new('RGB',(500,225),(30,33,38));d=ImageDraw.Draw(tile);tile.paste(crop,(40,100-crop.height//2));d.text((8,8),f'{n} input foreground W/H={ratio:.3f}',fill='white');canvas.paste(tile,(10,y))
 for col,(label,p) in enumerate([('old33k',r/'results_infer/data_test_stage1_lora_33k_stage2_lora_20k'/f'{n}.glb'),('axis10 20k',r/'results_infer'/tag/f'{n}.glb')],start=1):
  scene=trimesh.load(p,force='scene');mesh=scene.to_geometry();v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);e=np.asarray(scene.extents);center=np.asarray(scene.bounds).mean(0)
  xy=np.stack(((v-center)[:,2]*420+250,110-(v-center)[:,1]*420),axis=1);pixels=np.rint(xy).astype(int);order=np.argsort(v[f].mean(1)[:,0])[::-1];g=(80+145*np.abs(np.asarray(mesh.face_normals)[:,0])).astype(int)
  tile=Image.new('RGB',(500,225),(30,33,38));d=ImageDraw.Draw(tile)
  for i in order:
   poly=pixels[f[i]]
   if (poly[:,0]<0).all() or (poly[:,0]>=500).all() or (poly[:,1]<0).all() or (poly[:,1]>=225).all():continue
   value=int(g[i]);d.polygon([tuple(x) for x in poly],fill=(value,value,value))
  d.text((8,8),f'{n} {label} z/y={e[2]/e[1]:.3f} x/y={e[0]/e[1]:.3f}',fill='white');canvas.paste(tile,(10+col*510,y))
  print('PREVIEW',n,label,flush=True)
canvas.save(out);print(out,flush=True)
