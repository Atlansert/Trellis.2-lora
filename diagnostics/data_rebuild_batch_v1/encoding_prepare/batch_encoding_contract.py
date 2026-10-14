"""No model/GPU imports: encoding inputs must pass render review first."""
import hashlib,json,os
from pathlib import Path
import numpy as np
IDS={'0086bbab37559442','008ae427b2606cc8','015de7a65b71fbc2','0390f705ac242a39','1208e4cc0c30448a','1f7cc1ae1cec73f5'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def atomic(p,data):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 if tmp.exists():raise FileExistsError(tmp)
 with open(tmp,'x') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def bound_file(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe artifact path')
 p=Path(root)/p
 if not p.is_file():raise FileNotFoundError(p)
 return p
def load_geometry(path):
 with np.load(path,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
 if v.ndim!=2 or v.shape[1]!=3 or len(v)<3 or not np.isfinite(v).all():raise ValueError('invalid vertices')
 if f.ndim!=2 or f.shape[1]!=3 or len(f)<1 or not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(v):raise ValueError('invalid faces')
 if v.min()<-.500001 or v.max()>.500001 or np.argmax(np.ptp(v,axis=0))!=1:raise ValueError('geometry not normalized common raw lengthY')
 return v.astype(np.float32),f.astype(np.int64)
def validate_batch(path):
 path=Path(path).resolve();review=json.loads(path.read_text())
 if review.get('schema')!='trellis.batch-render-validation.v1' or review.get('passed') is not True or review.get('approved_for_encoding') is not True:raise ValueError('explicit parent render approval required')
 rows=review.get('assets',review.get('objects',[]))
 rows=[dict(r,render_dir=r.get('render_dir',r.get('render_directory'))) for r in rows]
 if len(rows)!=6 or {r['object_id'] for r in rows}!=IDS:raise ValueError('exact six-asset scope required')
 for row in rows:
  root=Path(row['render_dir'])
  if not root.is_absolute() or '..' in root.parts:raise ValueError('absolute safe render directory required')
  mp=root/'manifest.json';cp=root/'COMPLETE.json';m=json.loads(mp.read_text());c=json.loads(cp.read_text())
  if sha(mp)!=row['render_manifest_sha256'] or c.get('manifest_sha256')!=sha(mp) or c.get('views')!=8:raise ValueError('render completion checksum mismatch')
  if m.get('status')!='success' or len(m.get('views',[]))!=8 or [x['index'] for x in m['views']]!=list(range(8)):raise ValueError('incomplete render views')
  if m.get('training_approved') is not False or not m.get('all_reviewed_source_meshes_retained'):raise ValueError('unsafe render contract')
  if m.get('options',{}).get('resolution')!=512:raise ValueError('expected actual512 render')
  if sha(m['source_path'])!=m['source_sha256'] or sha(m['review_path'])!=m['review_sha256']:raise ValueError('source or asset review changed')
  for view in m['views']:
   if sha(bound_file(root,view['file']))!=view['sha256']:raise ValueError('render image changed')
  geom=bound_file(root,m['geometry']['path'])
  if sha(geom)!=m['geometry']['sha256']:raise ValueError('geometry checksum mismatch')
  load_geometry(geom)
  row['_render_manifest']=m;row['_geometry_path']=str(geom)
 return review,rows
def fresh_or_resume(output,batch_path):
 p=Path(output).resolve();fingerprint=sha(batch_path)
 if p.exists():
  marker=p/'INPUT.json'
  if not marker.is_file() or json.loads(marker.read_text()).get('batch_validation_sha256')!=fingerprint:raise FileExistsError('output exists without matching input contract')
 else:
  p.mkdir(parents=True);atomic(p/'INPUT.json',{'batch_validation_path':str(Path(batch_path).resolve()),'batch_validation_sha256':fingerprint,'training_approved':False})
 return p
def stage_done(folder,stage,input_sha):
 marker=folder/(stage+'_COMPLETE.json')
 if not marker.exists():return False
 rec=json.loads(marker.read_text())
 if rec.get('input_geometry_sha256')!=input_sha:raise ValueError('completed stage input changed')
 result=rec.get('result',{})
 for checkpoint in [result.get('checkpoint'),result.get('encoder'),result.get('decoder')]:
  if checkpoint is None:continue
  prefix=checkpoint['prefix']
  if sha(prefix+'.json')!=checkpoint['config_sha256'] or sha(prefix+'.safetensors')!=checkpoint['weight_sha256']:raise ValueError('completed stage model dependency changed')
 for name,digest in rec['files'].items():
  if sha(bound_file(folder,name))!=digest:raise ValueError('completed stage artifact changed')
 return True

def stage_commit(folder,stage,input_sha,files,result):
 for p in files:
  if not Path(p).is_file():raise FileNotFoundError(p)
 atomic(folder/(stage+'_COMPLETE.json'),{'stage':stage,'input_geometry_sha256':input_sha,'files':{Path(p).name:sha(p) for p in files},'result':result,'training_approved':False})

def assert_new(paths):
 if any(Path(p).exists() for p in paths):raise FileExistsError('partial stage outputs retained; use new output version instead of overwriting')

def device_guard(stage):
 value=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if stage in ('encode','decode') and value!='6':raise ValueError('GPU6 only for shape stages')
 if stage=='ss' and value!='':raise ValueError('SS encode stage CPU only')
