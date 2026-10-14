import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
root=Path('/data/lzx/TRELLIS.2');base=root/'diagnostics/remediation_20261013';out=root/'diagnostics/quality_followup_20261013/semantic_anchors'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((base/'shape_flow_compatibility/manifest.json').read_text());records=[]
canvas=Image.new('RGB',(1600,620),(24,27,31));draw=ImageDraw.Draw(canvas)
draw.text((8,8),'Camera-consistent +rawX view: screen-right=rawY=-formalZ; screen-up=rawZ=formalY. No best-view fitting.',fill='white')
draw.text((8,26),'Cyan roof / orange lower wheel band / red nose are PROGRAM CANDIDATES, not measured ground-truth keypoints.',fill='white')
for row,obj in enumerate(['0086bbab37559442','0fab4af3f9a73bdd']):
 y=55+row*280
 source=base/'common_frame_pilot'/obj/'proper_geometry.npz';z=np.load(source);gtv=z['vertices'];gtf=z['faces'];bounds=np.stack([gtv.min(0),gtv.max(0)]);length=np.ptp(gtv,axis=0)[1]
 # Per-end hull sections estimate tapered nose: thin endpoint vs wide box tail.
 profiles=[]
 for sign in [-1,1]:
  mask=(gtv[:,1]<bounds[0,1]+.08*length) if sign<0 else (gtv[:,1]>bounds[1,1]-.08*length)
  vv=gtv[mask];profiles.append({'end_rawY_sign':sign,'height_span':float(np.ptp(vv[:,2])),'width_span':float(np.ptp(vv[:,0])),'vertices':len(vv)})
 nose=min(profiles,key=lambda x:x['height_span']*x['width_span'])['end_rawY_sign']
 rec={'object':obj,'geometry_source_sha256':sha(source),'nose_candidate_rawY_end':nose,'end_sections':profiles,'cases':[],'camera_assumption':'current renderer yaw; historical transform matrices are placeholders'}
 rr=next(x for x in manifest['results'] if x['object']==obj and x['tag']=='official');im=Image.open(rr['outputs']['condition']['path']).convert('RGB');im.thumbnail((390,245));canvas.paste(im,(5,y+15));draw.text((5,y),obj+' condition',fill='white')
 for col,(tag,v,f,p) in enumerate([('GT',gtv,gtf,source)]+[(tag,np.load(base/'shape_flow_compatibility'/f'{obj}_{tag}'/'raw_geometry.npz')['vertices'],np.load(base/'shape_flow_compatibility'/f'{obj}_{tag}'/'raw_geometry.npz')['faces'],base/'shape_flow_compatibility'/f'{obj}_{tag}'/'raw_geometry.npz') for tag in ['official','fixed_shape20k']],start=1):
  tile=Image.new('RGB',(395,245),(29,32,36));d=ImageDraw.Draw(tile);center=bounds.mean(0);xy=np.stack([195+(v[:,1]-center[1])*370,135-(v[:,2]-center[2])*370],axis=1);pix=np.rint(xy).astype(int)
  # Full triangle binary silhouette, independent of tessellation.
  for tri in f:
   poly=pix[tri];d.polygon([tuple(q) for q in poly],fill=(175,180,184))
  bb=np.stack([v.min(0),v.max(0)]);sp=np.ptp(v,axis=0);cx=195;top=135-(bb[1,2]-center[2])*370;bottom=135-(bb[0,2]-center[2])*370
  d.line([(10,top),(380,top)],fill=(55,220,240),width=2);d.line([(10,bottom),(380,bottom)],fill=(240,170,65),width=2)
  nx=195+((bb[1,1] if nose>0 else bb[0,1])-center[1])*370;d.line([(nx,60),(nx,200)],fill=(240,70,80),width=2)
  d.text((8,8),tag+' +rawY right, +rawZ up',fill='white');d.text((8,220),f'height={sp[2]:.4f}; nose candidate {nose:+d}Y',fill='white');canvas.paste(tile,(col*400,y+15))
  # Area-unbiased sampled surface wheel band longitudinal distribution.
  vv=v[f];area=np.linalg.norm(np.cross(vv[:,1]-vv[:,0],vv[:,2]-vv[:,0]),axis=1)/2;cent=vv.mean(1);band=cent[:,2]<bb[0,2]+.2*sp[2];hist=np.histogram(cent[band,1],bins=np.linspace(bb[0,1],bb[1,1],21),weights=area[band])[0]
  rec['cases'].append({'tag':tag,'source_sha256':sha(p),'raw_bounds':bb.tolist(),'raw_spans':sp.tolist(),'lower_20pct_height_band_area_by_length_bins':hist.tolist(),'upper_roof_z_candidate':float(bb[1,2]),'lower_wheel_z_candidate':float(bb[0,2]),'semantic_labels_verified':False})
 records.append(rec);print('DONE',obj,profiles,'nose',nose,flush=True)
canvas.save(out/'semantic_camera_consistent.png');result={'schema':'trellis.semantic-anchor-diagnostic.v1','training_approved':False,'camera_screen_axes':{'position_raw':[1,0,0],'forward_raw':[-1,0,0],'up_raw':[0,0,1],'right_raw':[0,1,0],'right_formal':[0,0,-1]},'mirror_claim_supported':False,'historical_extrinsics_verified':False,'records':records};(out/'anchors.json').write_text(json.dumps(result,indent=2));print('OUTPUT',out,flush=True)
