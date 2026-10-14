from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
R=Path('/data/lzx/TRELLIS.2');O=R/'diagnostics/remediation_20261013/shape_flow_compatibility';M=json.loads((O/'manifest.json').read_text());assert M['status']=='success'
def load(path):
 with np.load(path) as a:return a['vertices'].astype(np.float64),a['faces'].astype(np.int64)
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(v,f):
 t=v[f];a=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2;c=t.mean(1);w=a/a.sum();mu=(w[:,None]*c).sum(0);d=c-mu;cov=np.einsum('n,ni,nj->ij',w,d,d);u=t-c[:,None,:];cov+=np.einsum('n,nki,nkj->ij',w,u,u)/12;e,q=np.linalg.eigh(cov);axis=q[:,-1];span=np.ptp(v,axis=0)
 # formal x,y,z=(rawx,rawz,-rawy)
 axisf=np.array([axis[0],axis[2],-axis[1]]);sf=span[[0,2,1]]
 return {'finite':bool(np.isfinite(v).all()),'vertices':len(v),'faces':len(f),'surface_area':float(a.sum()),'degenerate_faces':int((a==0).sum()),'raw_extents':span.tolist(),'formal_extents_xyz':sf.tolist(),'formal_z_over_y':float(sf[2]/sf[1]),'formal_x_over_y':float(sf[0]/sf[1]),'pca_method':'exact triangle surface integral covariance (within triangle variance included)','pca_axis_formal':axisf.tolist(),'pca_major_to_formal_z_deg':float(np.degrees(np.arccos(np.clip(abs(axisf[2]),0,1)))),'pca_eigenvalues':e.tolist()}
def sample(v,f,seed=4242,n=20000):
 rng=np.random.default_rng(seed);t=v[f];w=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1);i=rng.choice(len(f),n,p=w/w.sum());u=rng.random((n,2));s=np.sqrt(u[:,0]);b=np.column_stack([1-s,s*(1-u[:,1]),s*u[:,1]]);return np.einsum('ni,nij->nj',b,t[i])
def distance(a,b):
 da=cKDTree(b).query(a,workers=8)[0];db=cKDTree(a).query(b,workers=8)[0]
 return {'sample_count_each':len(a),'sampling_seed':4242,'registration':'none; no ICP, re-centering, scaling or best rotation','method':'bidirectional nearest sampled surface point, not exact triangle distance','pred_to_gt_mean':float(da.mean()),'gt_to_pred_mean':float(db.mean()),'symmetric_mean':float((da.mean()+db.mean())/2),'rms_symmetric':float(np.sqrt((np.square(da).mean()+np.square(db).mean())/2)),'pred_to_gt_p95':float(np.quantile(da,.95)),'gt_to_pred_p95':float(np.quantile(db,.95)),'symmetric_max':float(max(da.max(),db.max()))}
def panel(v,f,label,oblique=False):
 im=Image.new('RGB',(470,250),(30,33,38));d=ImageDraw.Draw(im);d.text((8,6),label,fill='white');vf=np.column_stack([v[:,0],v[:,2],-v[:,1]])
 if oblique:
  hor=np.array([.45,0,.893]);ver=np.array([-.2,.975,.10]);ver-=hor*np.dot(hor,ver);ver/=np.linalg.norm(ver);depth=np.cross(hor,ver)
 else:hor=np.array([0,0,1]);ver=np.array([0,1,0]);depth=np.array([1,0,0])
 # fixed 420 pixels per normalized unit across all GT and predictions; no per-mesh fit
 pts=np.column_stack([235+420*(vf@hor),153-420*(vf@ver)]);dep=vf@depth
 step=max(1,len(f)//80000);ff=f[::step];z=dep[ff].mean(1);idx=np.argsort(z)
 tri=vf[ff];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(n,axis=1);light=np.array([.4,.8,.5]);light/=np.linalg.norm(light);lum=np.clip(abs(n@light)/np.maximum(norm,1e-15),0,1)
 for i in idx:
  poly=pts[ff[i]].reshape(-1).tolist();g=int(70+165*lum[i]);d.polygon(poly,fill=(g,g,g))
 return im
report={'schema':'trellis.shape-flow-compatibility-metrics.v1','runtime_manifest_sha256':hashfile(O/'manifest.json'),'training_approved':False,'formal_acceptance_eligible':False,'single_variable_checks':M['single_variable_checks'],'cases':[],'comparison_scope':'Only 2 assets with GT-provided correct support, not SS-flow predictions or 548-object generalization. No texture or formal remesh.'}
canvas=Image.new('RGB',(1880,530),(20,22,26));obl=canvas.copy()
for row,obj in enumerate(['0086bbab37559442','0fab4af3f9a73bdd']):
 v,f=load(R/'diagnostics/remediation_20261013/common_frame_pilot'/obj/'proper_geometry.npz');gt=sample(v,f);basecase={'object':obj,'ground_truth':metrics(v,f)}
 for cv,ob in [(canvas,False),(obl,True)]:
  image=Image.open(O/f'{obj}_official/condition_black.png').convert('RGB');image.thumbnail((440,210));cv.paste(image,(15+(440-image.width)//2,row*260+30));ImageDraw.Draw(cv).text((10,row*260+7),obj+' condition native alpha',fill='white');cv.paste(panel(v,f,'GT proper geometry; fixed normalized scale',ob),(470,row*260))
 for col,tag in [(2,'official'),(3,'fixed_shape20k')]:
  vp,fp=load(O/f'{obj}_{tag}/raw_geometry.npz');m=metrics(vp,fp);m['chamfer_sampling']=distance(sample(vp,fp),gt);basecase[tag]=m
  for cv,ob in [(canvas,False),(obl,True)]:cv.paste(panel(vp,fp,f'{tag} z/y={m["formal_z_over_y"]:.3f}; tilt={m["pca_major_to_formal_z_deg"]:.3f}deg',ob),(col*470,row*260))
 basecase['fixed_minus_official_symmetric_mean']=basecase['fixed_shape20k']['chamfer_sampling']['symmetric_mean']-basecase['official']['chamfer_sampling']['symmetric_mean'];report['cases'].append(basecase)
ImageDraw.Draw(canvas).text((10,517),'Fixed formal ZY projection; +Y up; same scale; no axis correction/ICP. Sparse face painter is approximate, not complete GLB acceptance.',fill='white');ImageDraw.Draw(obl).text((10,517),'Fixed shared oblique projection; same scale; geometry only. Do not infer semantic nose/roof correctness from PCA or Chamfer alone.',fill='white')
canvas.save(O/'fixed_profile.png');obl.save(O/'fixed_oblique.png');report['images']={x:hashfile(O/x) for x in ['fixed_profile.png','fixed_oblique.png']}
report['decision']='UNVERIFIED: quantitative distances and geometry profile require interpretation; no automatic training approval or geometry pass.'
p=O/'metrics.json';assert not p.exists();p.write_text(json.dumps(report,indent=2));print(json.dumps(report['cases'],indent=2));print('COMPAT_CPU_METRICS_COMPLETE')
