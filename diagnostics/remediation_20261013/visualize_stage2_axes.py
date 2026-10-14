import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
r=Path('training_data/multiview_1186');out=Path('diagnostics/remediation_20261013');ids=['0086bbab37559442','0fab4af3f9a73bdd'];canvas=Image.new('RGB',(1500,650),(25,28,33));d=ImageDraw.Draw(canvas);d.text((10,8),'Stage2: SAME asset native-alpha image / canonical SS support / shape TRAINING support. Fixed axes, no correction.',fill='white');records=[]
for row,obj in enumerate(ids):
 m=json.loads((out/'canonical_contract_v2_native_staged/manifest.json').read_text());ob=next(x for x in m['objects'] if x['object']==obj);v=ob['candidate_target_view'];rr=r/'renders_cond_t'/obj;f=json.loads((rr/'transforms.json').read_text())['frames'][v]
 im=Image.open(rr/f['file_path']).convert('RGBA');a=np.asarray(im.getchannel('A'));ys,xs=np.nonzero(a>204);crop=im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1));rgb=Image.new('RGB',crop.size,'black');rgb.paste(crop,mask=crop.getchannel('A'));rgb.thumbnail((440,220));y=45+row*300;canvas.paste(rgb,(25,y+90));d.text((15,y),obj+' native condition view '+str(v),fill='white')
 ss=np.load(r/'ss_view_voxels_canon_v4'/obj/f'{v:03d}.npz')['coords'][:,[2,0,1]].astype(int);shape=np.load(r/'shape_latents/shape_enc_next_dc_f16c32_fp16_1024'/f'{obj}.npz')['coords'].astype(int)
 for col,(title,c) in enumerate([('SS target voxels',ss),('shape TRAINING coords',shape)],start=1):
  x=col*500;d.text((x+8,y),title+' spans='+str((np.ptp(c,axis=0)+1).tolist()),fill='white')
  # Expose native axis identity: horizontal plot is axis1, vertical is axis0.
  for axh,axv,xoff,label in [(1,0,0,'dim1 horizontal / dim0 vertical'),(1,2,245,'dim1 horizontal / dim2 vertical')]:
   d.rectangle((x+8+xoff,y+35,x+242+xoff,y+269),outline=(70,80,90))
   for pt in c:
    px=x+10+xoff+int(pt[axh]*3.55);py=y+265-int(pt[axv]*3.55);d.rectangle((px,py,px+3,py+3),fill=(95,190,230))
   d.text((x+8+xoff,y+276),label,fill='white')
 z=pickle.load(open(r/'mesh_dumps'/f'{obj}.pickle','rb'));verts=np.concatenate([o['vertices'] for o in z['objects']],0);export=verts.copy();export[:,1],export[:,2]=export[:,2],-export[:,1]
 records.append({'object':obj,'image_view':v,'ss_spans':(np.ptp(ss,axis=0)+1).tolist(),'shape_spans':(np.ptp(shape,axis=0)+1).tolist(),'mesh_dump_extents':np.ptp(verts,axis=0).tolist(),'mesh_dump_formal_export_rotation_extents':np.ptp(export,axis=0).tolist(),'dump_long_axis':int(np.argmax(np.ptp(verts,axis=0))),'dump_after_glb_rotation_long_axis':int(np.argmax(np.ptp(export,axis=0))),'not_model_output':True})
canvas.save(out/'stage2_real_asset_axes.png');(out/'stage2_real_asset_axes.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
