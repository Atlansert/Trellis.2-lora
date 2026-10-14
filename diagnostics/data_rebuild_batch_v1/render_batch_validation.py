"""CPU independent batch gate; never builds models or approves training."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('/data/lzx/TRELLIS.2');b=root/'diagnostics/data_rebuild_batch_v1';reviews=json.loads((b/'asset_review/review.json').read_text());out=b/'batch_render_validation.json'
if out.exists():raise FileExistsError(out)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
for e in reviews['selected']:
 oid=e['object_id'];d=b/'renders'/f'{oid}_cpu_v1';mp=d/'manifest.json';m=json.loads(mp.read_text());commit=json.loads((d/'COMPLETE.json').read_text());assert sha(mp)==commit['manifest_sha256'];assert m['status']=='success' and len(m['views'])==8 and commit['views']==8
 assert sha(e['review_path'])==m['review_sha256']==e['review_sha256'];assert sha(e['source']['path'])==m['source_sha256']==e['source']['sha256']
 geometry=d/m['geometry']['path'];assert sha(geometry)==m['geometry']['sha256'];g=np.load(geometry,allow_pickle=False);v=g['vertices'];f=g['faces'];assert np.isfinite(v).all() and v.ndim==2 and v.shape[1]==3 and len(f)>0 and f.max()<len(v) and f.min()>=0;assert np.argmax(np.ptp(v,axis=0))==1
 views=[]
 for r in m['views']:
  im=d/r['file'];assert sha(im)==r['sha256']
  with Image.open(im) as image:assert image.mode=='RGBA' and image.size==(512,512);a=np.asarray(image)[:,:,3]
  strong=a>204;assert strong.any() and a.min()==0 and not (a==255).all();ys,xs=np.nonzero(strong);assert xs.min()>0 and xs.max()<511 and ys.min()>0 and ys.max()<511
  C=np.asarray(r['matrix_world_camera_to_world']);W=np.asarray(r['matrix_world_to_camera']);K=np.asarray(r['intrinsic_K_cv']);assert np.allclose(C@W,np.eye(4),atol=1e-6);assert np.allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=1e-6);assert abs(np.linalg.det(C[:3,:3])-1)<1e-6
  pts=np.asarray(r['projection_audit_world_points']);cam=np.column_stack((pts,np.ones(len(pts))))@W.T;depth=-cam[:,2];assert (depth>0).all();pixels=np.column_stack((K[0,0]*cam[:,0]/depth+K[0,2],K[1,2]-K[1,1]*cam[:,1]/depth));err=float(abs(pixels-np.asarray(r['projection_audit_pixels_cv'])).max());assert err<1e-5
  if r['index']==0:assert np.allclose(C[:3,0],[0,1,0],atol=1e-6) and np.allclose(C[:3,1],[0,0,1],atol=1e-6)
  views.append({'index':r['index'],'alpha_strong_pixels':int(strong.sum()),'bbox':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],'reprojection_max_error_pixels':err,'image_sha256':sha(im)})
 rows.append({'object_id':oid,'render_directory':str(d),'render_manifest_sha256':sha(mp),'geometry_sha256':sha(geometry),'views':views,'passed':True})
result={'schema':'trellis.batch-render-validation.v1','passed':True,'approved_for_encoding':True,'training_approved':False,'objects':rows,'object_ids':[x['object_id'] for x in rows],'assets':[{'object_id':x['object_id'],'render_dir':x['render_directory'],'render_manifest_sha256':x['render_manifest_sha256']} for x in rows],'images':48,'scope':'all48hashes+RGBAalpha+noedgeclipping+worldgeometryfinite+rawlengthY+actualcamera/inverse/Kreprojection; perasset image visual review separately required beforetraining','float32_matrix_tolerance':1e-6};out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='objects'},indent=2))
