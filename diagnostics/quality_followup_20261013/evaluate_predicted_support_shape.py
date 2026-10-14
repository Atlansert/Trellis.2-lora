import os,sys,json,hashlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
R=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(R));os.chdir(R)
import numpy as np,trimesh
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from scripts.utils.evaluate_fixed_frame import mesh_metrics
B=R/'diagnostics/quality_followup_20261013/predicted_support_shape';m=json.loads((B/'manifest.json').read_text());assert m['status']=='success'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def samples(v,f,n=20000):
 t=v[f];areas=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2;rng=np.random.default_rng(4242);ids=rng.choice(len(f),n,p=areas/areas.sum());a=np.sqrt(rng.random(n));b=rng.random(n);return (1-a[:,None])*t[ids,0]+(a*(1-b))[:,None]*t[ids,1]+(a*b)[:,None]*t[ids,2]
def stats(v,f):
 formal=v[:,[0,2,1]]*np.array([1,1,-1]);s=mesh_metrics(trimesh.Trimesh(formal,f,process=False));sections=[]
 for x in np.linspace(-.45,.45,10):
  vv=v[np.abs(v[:,1]-x)<.025];sections.append({'raw_length_center':float(x),'width':float(np.ptp(vv[:,0])) if len(vv) else None,'height':float(np.ptp(vv[:,2])) if len(vv) else None})
 s['raw_body_crosssections']=sections;return s
src=R/'diagnostics/remediation_20261013/common_frame_pilot/0086bbab37559442/proper_geometry.npz'
with np.load(src) as z:gv=z['vertices'];gf=z['faces']
gt=stats(gv,gf);gpts=samples(gv,gf);records=[];canvas=Image.new('RGB',(2400,650),(22,25,29));d=ImageDraw.Draw(canvas);d.text((10,8),'Predicted SS -> shape factorial. Camera +rawX: right+rawY/up+rawZ. Full triangle silhouette, same scale, NO alignment.',fill='white');d.text((10,28),'Train GT only 0086; heldout004 has no identity/GT. Geometry-only diagnostics, NOT formal GLB acceptance.',fill='white')
for row,name in enumerate(['train0086','004']):
 y=60+row*285
 im=Image.open(B/f'condition_official_{name}.png').convert('RGB');im.thumbnail((390,235));canvas.paste(im,(5,y+30));d.text((5,y),name+' condition',fill='white')
 center=np.stack([gv.min(0),gv.max(0)]).mean(0) if name=='train0086' else np.zeros(3)
 sets=[('GT',gv,gf)] if name=='train0086' else []
 for rec in [x for x in m['cases'] if x['input']==name]:
  p=Path(rec['outputs']['raw_geometry.npz']['path']);assert sha(p)==rec['outputs']['raw_geometry.npz']['sha256']
  with np.load(p) as z:v=z['vertices'];f=z['faces']
  result={'input':name,'ss':rec['ss'],'shape':rec['shape'],'artifact_sha256':sha(p),'metrics':stats(v,f)}
  if name=='train0086':
   pts=samples(v,f);a=cKDTree(gpts).query(pts)[0];b=cKDTree(pts).query(gpts)[0];result['sampled_surface_chamfer']={'symmetric_mean':float((a.mean()+b.mean())/2),'pred_to_gt_mean':float(a.mean()),'gt_to_pred_mean':float(b.mean()),'p95_pred_to_gt':float(np.percentile(a,95)),'samples_each':20000,'seed':4242,'method':'nearest sampled point, not exact triangle distance; NO ICP/rotation/scaling'}
  else:result['no_gt']='no asset identity; no fidelity metric'
  records.append(result);sets.append((rec['ss']+'/'+rec['shape'],v,f));print('METRIC',name,rec['ss'],rec['shape'],result['metrics']['extents_xyz'],result.get('sampled_surface_chamfer',{}).get('symmetric_mean'),flush=True)
 for col,(tag,v,f) in enumerate(sets,start=1):
  tile=Image.new('RGB',(395,250),(29,32,36));draw=ImageDraw.Draw(tile);pix=np.rint(np.stack([195+(v[:,1]-center[1])*370,135-(v[:,2]-center[2])*370],axis=1)).astype(int)
  for tri in f:draw.polygon([tuple(pt) for pt in pix[tri]],fill=(170,180,188))
  draw.text((8,8),tag,fill='white');draw.text((8,232),'height rawZ='+str(round(float(np.ptp(v[:,2])),4)),fill='white');canvas.paste(tile,(col*400,y+20))
canvas.save(B/'same_camera_full_silhouettes.png')
r={'schema':'trellis.predicted-support-shape-metrics.v1','manifest_sha256':sha(B/'manifest.json'),'gt_source_sha256':sha(src),'train_gt_metrics':gt,'records':records,'ss_samples':m['ss_samples'],'formal_acceptance_eligible':False,'training_continued':False,'limits':'8 cases two inputs; no texture/remesh; heldout noGT; full raster no sparse-face skipping; sampled Chamfer not exact surface'}
(B/'metrics.json').write_text(json.dumps(r,indent=2));(B/'METRICS_COMPLETE.json').write_text(json.dumps({'metrics_sha256':sha(B/'metrics.json'),'image_sha256':sha(B/'same_camera_full_silhouettes.png')},indent=2));print('METRICS_COMPLETE',flush=True)
