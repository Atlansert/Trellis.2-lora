import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,trimesh
from PIL import Image,ImageDraw
r=Path('/data/lzx/TRELLIS.2');b=r/'diagnostics/cohort_training_followup_v1';out=b/'formal002_50_vs200.png'
if out.exists():raise FileExistsError(out)
canvas=Image.new('RGB',(1600,580),(27,30,35));d=ImageDraw.Draw(canvas);d.text((8,6),'002 original GLB coordinates, fixed ZY side / ZX top, same normalized length scale, all triangles, no alignment.',fill='white')
paths=[('old33k',r/'results_infer/data_test_stage1_lora_33k_stage2_lora_20k/002.glb'),('rebuilt50',r/'diagnostics/data_rebuild_batch_v1/formal_glb_ss_live50/002/002.glb'),('rebuilt200',b/'formal_step200/002/002.glb')]
im=Image.open(r/'data_test/002.png').convert('RGB');im.thumbnail((380,380));canvas.paste(im,(10,60));d.text((10,40),'Input; no matched 3D GT',fill='white')
for col,(label,path) in enumerate(paths,1):
 scene=trimesh.load(path,force='scene',process=False);mesh=scene.to_geometry();v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);center=scene.bounds.mean(0)
 for row,(axes,text) in enumerate([((2,1),'side ZY'),((2,0),'top ZX')]):
  x=col*400;y=55+row*255;tile=Image.new('RGB',(390,235),(33,36,41));draw=ImageDraw.Draw(tile);pts=np.column_stack((190+(v[:,axes[0]]-center[axes[0]])*350,120-(v[:,axes[1]]-center[axes[1]])*350));pts=np.rint(pts).astype(int)
  for face in f:
   p=pts[face];draw.polygon([tuple(q) for q in p],fill=(175,181,186))
  draw.text((5,5),label+' '+text,fill='white');canvas.paste(tile,(x,y))
 print('PLANNED_VIEW_DONE',label,flush=True)
canvas.save(out);print(out,flush=True)
