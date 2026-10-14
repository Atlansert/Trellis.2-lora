"""CPU-only gates; actual models imported only by explicit execution stages."""
import hashlib,json,os
from pathlib import Path
import numpy as np
IDS={'0086bbab37559442','008ae427b2606cc8','015de7a65b71fbc2','0390f705ac242a39','1208e4cc0c30448a','1f7cc1ae1cec73f5'}
CHANGED={'1208e4cc0c30448a','1f7cc1ae1cec73f5'}
OLD=Path('/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/encoded_targets_v1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);t=p.with_name(p.name+'.tmp')
 if t.exists():raise FileExistsError(t)
 with open(t,'x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def artifact(r):
 p=Path(r['path'])
 if not p.is_absolute() or '..' in p.parts or not p.is_file() or sha(p)!=r['sha256']:raise ValueError('artifact changed/unsafe')
 return p
def matrix_check(r,changed):
 R=np.asarray(r,dtype=np.float64);expected=np.diag([-1,-1,1]) if changed else np.eye(3)
 if R.shape!=(3,3) or not np.allclose(R,expected,atol=1e-12) or not np.allclose(R.T@R,np.eye(3),atol=1e-12) or abs(np.linalg.det(R)-1)>1e-12:raise ValueError('proper declared rotation required')
 return R
def load_geometry(p):
 with np.load(p,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or len(v)<3 or f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(v):raise ValueError('invalid geometry')
 if v.min()<-.500001 or v.max()>.500001 or np.argmax(np.ptp(v,axis=0))!=1:raise ValueError('wrong normalized frame')
 return v.astype(np.float32),f.astype(np.int64)
def checkpoint_check(c):
 if sha(c['prefix']+'.json')!=c['config_sha256'] or sha(c['prefix']+'.safetensors')!=c['weight_sha256']:raise ValueError('checkpoint dependency changed')
def stage_done(folder,stage,gh):
 p=folder/(stage+'_COMPLETE.json')
 if not p.exists():return False
 x=json.loads(p.read_text())
 if x['input_geometry_sha256']!=gh:raise ValueError('completed input changed')
 for name,h in x['files'].items():
  if Path(name).name!=name or sha(folder/name)!=h:raise ValueError('completed output changed')
 for key in ['checkpoint','encoder','decoder']:
  if x['result'].get(key):checkpoint_check(x['result'][key])
 return True
def stage_commit(folder,stage,gh,files,result):
 p=folder/(stage+'_COMPLETE.json')
 if p.exists():raise FileExistsError(p)
 atomic(p,{'stage':stage,'input_geometry_sha256':gh,'files':{x.name:sha(x) for x in files},'result':result,'training_approved':False})
def assert_new(paths):
 if any(x.exists() for x in paths):raise FileExistsError('partial outputs retained')
def device_guard(stage):
 if stage in ['encode','decode'] and os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise ValueError('GPU6 explicit execution only')
 if stage in ['ss','finalize'] and os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('CPU stage required')
def validate_batch(path):
 path=Path(path).resolve();m=json.loads(path.read_text());c=json.loads((path.parent/'COMPLETE.json').read_text())
 if c.get('manifest_sha256')!=sha(path) or m.get('schema')!='trellis.nose-canonical-geometry-candidate.v1' or m.get('training_approved') is not False or m.get('condition_images_changed') is not False:raise ValueError('candidate contract invalid')
 parent=path.parent/'PARENT_GEOMETRY_REVIEW.json';review=json.loads(parent.read_text())
 if not review.get('passed_geometry_candidate_review') or review.get('training_approved') is not False or {x['object_id'] for x in review['rows']}!=IDS:raise ValueError('parent geometry-only review missing')
 records=m['records']
 if len(records)!=6 or {x['object_id'] for x in records}!=IDS:raise ValueError('six scope required')
 oldm=json.loads((OLD/'manifest.json').read_text());oldc=json.loads((OLD/'COMPLETE.json').read_text())
 if oldc['manifest_sha256']!=sha(OLD/'manifest.json'):raise ValueError('original encoded manifest changed')
 by={x['object_id']:x for x in oldm['assets']};rows=[]
 for r in records:
  obj=r['object_id'];changed=obj in CHANGED;R=matrix_check(r['matrix_row_oldraw_to_canonical'],changed);src=artifact(r['source_geometry']);artifact(r['source_condition']);sv,sf=load_geometry(src)
  if changed:
   geom=artifact(r['geometry_candidate']);v,f=load_geometry(geom)
   if not np.array_equal(f,sf) or not np.allclose(v,sv@R,atol=1e-7,rtol=0):raise ValueError('candidate not exact proper geometry transform')
  else:
   if r['geometry_candidate'] is not None or Path(r['identity_geometry_reference']).resolve()!=src.resolve():raise ValueError('identity must reuse exact original geometry')
   geom=src
   if by[obj]['geometry_sha256']!=sha(src):raise ValueError('identity encoded geometry mismatch')
   for stage in by[obj]['stages']:
    if not stage_done(OLD/obj,stage,sha(src)):raise ValueError('identity encoded stage invalid')
  rows.append({'object_id':obj,'changed':changed,'_geometry_path':str(geom),'geometry_sha256':sha(geom),'condition':r['source_condition'],'same_image_camera_to_world_newcanonical':r['same_image_camera_to_world_newcanonical'],'identity_encoded_reference':None if changed else str(OLD/obj),'split':r['split']})
 return {'candidate_manifest_sha256':sha(path),'parent_review_sha256':sha(parent),'original_encoded_manifest_sha256':sha(OLD/'manifest.json'),'condition_policy':'six unchanged cache images, target-pose canonicalization; no fabricated new render','training_approved':False},rows
def fresh_or_resume(output,path):
 p=Path(output).resolve();binding,_=validate_batch(path)
 if p.exists():
  if not (p/'INPUT.json').is_file() or json.loads((p/'INPUT.json').read_text())!=binding:raise FileExistsError('unrelated existing output')
 else:p.mkdir(parents=True);atomic(p/'INPUT.json',binding)
 return p
