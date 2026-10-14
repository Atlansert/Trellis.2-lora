"""Small rebuilt-data pilot, preparation only until independent parent approval."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
SCHEMA='trellis.six-render-ss-pilot.v1'
TRAIN=('0086bbab37559442','008ae427b2606cc8','015de7a65b71fbc2','1208e4cc0c30448a')
HOLDOUT=('0390f705ac242a39','1f7cc1ae1cec73f5')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def record(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}

def verify(r):
 p=Path(r['path'])
 if not p.is_absolute() or '..' in p.parts:raise ValueError('absolute nontraversing path required')
 if not p.is_file() or p.stat().st_size!=r['size_bytes'] or sha(p)!=r['sha256']:raise ValueError('artifact checksum/size changed')
 return p

def production_condition(rgba):
 if rgba.mode!='RGBA':raise ValueError('actual RGBA required')
 a=np.asarray(rgba);alpha=a[:,:,3];ys,xs=np.nonzero(alpha>204)
 if not len(xs) or np.all(alpha==255):raise ValueError('empty/opaque foreground')
 cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2
 size=int(max(xs.max()-xs.min(),ys.max()-ys.min()))
 if size<2:raise ValueError('degenerate crop')
 cropped=rgba.crop((cx-size//2,cy-size//2,cx+size//2,cy+size//2))
 arr=np.asarray(cropped).astype(np.float32)/255
 rgb=Image.fromarray((arr[:,:,:3]*arr[:,:,3:4]*255).astype(np.uint8))
 return rgb.resize((512,512),Image.Resampling.LANCZOS)

def finite_mean(p):
 with np.load(p,allow_pickle=False) as z:
  if 'mean' not in z:raise ValueError('mean missing')
  a=z['mean'].copy()
 if a.shape!=(8,16,16,16) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('finite fp32 8x16^3 required')
 return a

def fresh_output(p):
 p=Path(p).resolve()
 if p.exists():raise FileExistsError('no output overwrite or implicit resume')
 p.mkdir(parents=True);return p

def phase_steps(requested,allowed=50):
 if type(requested)!=int or requested!=50 or requested>allowed:raise ValueError('only first50step phase implemented; no automatic100continuation')
 return requested

def validate(manifest,approval=None,diagnostic=True):
 path=Path(manifest).resolve();m=json.loads(path.read_text());c=json.loads(path.with_name('COMPLETE.json').read_text())
 if c.get('manifest_sha256')!=sha(path) or m.get('schema')!=SCHEMA:raise ValueError('manifest identity invalid')
 if m.get('production_training_approved') is not False or m.get('target_frame')!='proper_raw_depth_length_up':raise ValueError('invalid frame/production claim')
 if tuple(m.get('train_objects',[]))!=TRAIN or tuple(m.get('internal_holdout_objects',[]))!=HOLDOUT:raise ValueError('split changed or holdout leak')
 if set(TRAIN)&set(HOLDOUT):raise ValueError('split overlap')
 rows=m['objects'];by={x['object_id']:x for x in rows}
 if len(by)!=6 or len(rows)!=6 or set(by)!=set(TRAIN+HOLDOUT):raise ValueError('exact6objects required')
 for r in m['dependencies'].values():verify(r)
 for obj,x in by.items():
  if x['split']!=('train' if obj in TRAIN else 'internal_holdout') or x.get('view_index')!=0:raise ValueError('wrong split or extra image views')
  for key in ['render_rgba','render_manifest','canonical_geometry','target','condition','source_review']:verify(x[key])
  with Image.open(x['condition']['path']) as im:
   if im.mode!='RGB' or im.size!=(512,512):raise ValueError('condition RGB512 required')
  finite_mean(x['target']['path'])
  render=json.loads(Path(x['render_manifest']['path']).read_text())
  if render.get('status')!='success' or render['raw_axes'].get('up')!='z' or render['raw_axes'].get('length')!='y':raise ValueError('render frame not approved raw contract')
 if not diagnostic:
  if approval is None:raise ValueError('independent parent approval required')
  a=json.loads(Path(approval).read_text())
  if a.get('schema')!='trellis.six-render-ss-approval.v1' or a.get('manifest_sha256')!=sha(path) or a.get('approved') is not True or a.get('diagnostic_only') is not True:raise ValueError('approval identity invalid')
  if tuple(a.get('train_objects',[]))!=TRAIN or tuple(a.get('internal_holdout_objects',[]))!=HOLDOUT:raise ValueError('approval split differs')
  phase_steps(a.get('first_phase_steps'),a.get('total_steps_limit',50))
  evidence=verify(a['target_validation']);gate=json.loads(evidence.read_text())
  if gate.get('schema')!='trellis.six-target-scoped-data-gate.v1' or gate.get('data_validation_passed') is not True or gate.get('approved_for_short_training') is not True or set(gate.get('objects',[]))!=set(TRAIN+HOLDOUT) or gate.get('approved_condition_views')!=[0] or gate.get('production_approved') is not False:raise ValueError('target validation not passed exact scope')
  proof=Path(gate.get('validation_path',''))
  if not proof.is_absolute() or not proof.is_file() or sha(proof)!=gate.get('validation_sha256'):raise ValueError('target validation proof checksum mismatch')
  for key in ('render_visual_review','split_evidence'):
   ep=verify(a[key]);doc=json.loads(ep.read_text())
   if key=='render_visual_review' and (doc.get('schema')!='trellis.render-batch-visual-review.v1' or doc.get('passed') is not True or set(doc.get('objects',[]))!=set(TRAIN+HOLDOUT) or doc.get('reviewed_condition_views')!=[0]):raise ValueError('render visual review not passed')
   if key=='split_evidence' and (doc.get('schema')!='trellis.small-cohort-split.v1' or doc.get('train')!=list(TRAIN) or doc.get('heldout')!=list(HOLDOUT) or doc.get('condition_views')!=[0] or doc.get('first_phase_steps')!=50 or doc.get('resume_to100_automatic') is not False):raise ValueError('split evidence does not match plan')
  if a.get('encoded_manifest_sha256')!=m['dependencies']['encoded_manifest']['sha256']:raise ValueError('encoded targets approval mismatch')
 return m,by

class ManifestDataset:
 value_range=(0,1);normalization=None;skip_snapshot=True
 def __init__(self,path,approval):
  self.path=path;self.approval=approval;self.m,self.by=validate(path,approval,False);self.objects=list(TRAIN)
 def __len__(self):return len(self.objects)
 def __getitem__(self,i):
  import torch
  x=self.by[self.objects[i]];cp=verify(x['condition']);tp=verify(x['target'])
  with Image.open(cp) as im:c=np.asarray(im,dtype=np.float32)/255
  return {'x_0':torch.from_numpy(finite_mean(tp)),'cond':torch.from_numpy(c.copy()).permute(2,0,1)}
 def visualize_sample(self,x):
  if isinstance(x,dict):x=x['x_0']
  return x[:,:3,:,:,x.shape[-1]//2]
 def __str__(self):return 'Rebuilt6AssetSS:train4,internal_holdout2,view000only; diagnostic, notproduction'
