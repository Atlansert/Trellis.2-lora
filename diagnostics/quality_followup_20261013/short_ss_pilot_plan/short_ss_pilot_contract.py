"""Two-candidate diagnostic only; no production registration or implicit approval."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def verify_artifact(rec):
 p=Path(rec['path'])
 if not p.is_absolute() or '..' in p.parts:raise ValueError('absolute nontraversing artifact required')
 if not p.is_file() or sha(p)!=rec['sha256']:raise ValueError('artifact checksum mismatch')
 return p

def validate(manifest_path,arm,approval_path=None,diagnostic_only=False):
 if arm not in ('legacy','proper'):raise ValueError('unknown arm')
 p=Path(manifest_path);m=json.loads(p.read_text());c=json.loads(p.with_name('COMPLETE.json').read_text())
 if c['manifest_sha256']!=sha(p) or m['schema']!='trellis.short-ss-pilot.v1':raise ValueError('manifest checksum/schema mismatch')
 if m.get('production_approved') is not False:raise ValueError('diagnostic manifest must not claim production approval')
 objs=m['objects'];names=[x['object'] for x in objs]
 if len(names)!=len(set(names)) or set(names)!={'0086bbab37559442','0fab4af3f9a73bdd'}:raise ValueError('candidate scope changed')
 approved=names
 if 'base' in m:verify_artifact(m['base'])
 if 'recipe' in m:verify_artifact(m['recipe'])
 if not diagnostic_only:
  if approval_path is None:raise ValueError('parent approval required before training')
  a=json.loads(Path(approval_path).read_text())
  if a.get('schema')!='trellis.short-ss-pilot-approval.v1' or a.get('manifest_sha256')!=sha(p) or a.get('approved') is not True or a.get('diagnostic_only') is not True:raise ValueError('invalid pilot approval')
  approved=a.get('objects',[])
  if not approved or not set(approved).issubset(names) or len(approved)!=len(set(approved)):raise ValueError('approval scope invalid')
  allowed=set(m.get('approval_scope_limit',names))
  if not set(approved).issubset(allowed) or any(o.get('execution_eligible') is False for o in objs if o['object'] in approved):raise ValueError('quarantined object cannot be approved by this plan')
  if not a.get('geometry_review_passed') or not a.get('semantic_review_passed'):raise ValueError('geometry/semantic review required')
  for key in ('geometry_evidence','semantic_evidence'):
   ep=verify_artifact(a.get(key,{}))
   evidence=json.loads(ep.read_text())
   if evidence.get('passed') is not True or not set(approved).issubset(evidence.get('objects',[])):raise ValueError('review evidence scope/passed mismatch')
  if a.get('max_steps') not in (100,200):raise ValueError('bounded step approval required')
 for obj in objs:
  if obj['target_frames']!={'legacy':'legacy_ss_down_length_depth','proper':'proper_raw_depth_length_up'}:raise ValueError('target frame labels wrong')
  cp=verify_artifact(obj['condition']);im=Image.open(cp)
  if im.mode!='RGB' or im.size!=(512,512):raise ValueError('condition must be cached RGB512')
  im.close()
  for tag in ('legacy','proper'):
   lp=verify_artifact(obj['targets'][tag])
   with np.load(lp,allow_pickle=False) as z:v=z['mean']
   if v.shape!=(8,16,16,16) or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('mean must be finite FP32 8x16x16x16')
 return m,[x for x in objs if x['object'] in approved]

class ManifestDataset:
 value_range=(0,1);normalization=None;skip_snapshot=True
 def __init__(self,path,arm,approval_path=None,diagnostic_only=False):
  self.manifest_path=path;self.arm=arm;self.approval=approval_path;self.diagnostic_only=diagnostic_only
  self.manifest,self.objects=validate(path,arm,approval_path,diagnostic_only)
 def __len__(self):
  # Existing trainer drop_last=True: a one-object approved scope must expose
  # two REPEATED draw slots for batch2, not invent a second asset.
  return max(2,len(self.objects))
 def __getitem__(self,i):
  import torch
  # Hash-verify mutable inputs at each access, including target and condition.
  o=self.objects[i % len(self.objects)];cp=verify_artifact(o['condition']);lp=verify_artifact(o['targets'][self.arm])
  with Image.open(cp) as im:c=np.asarray(im,dtype=np.float32)/255
  with np.load(lp,allow_pickle=False) as z:x=z['mean'].copy()
  if not np.isfinite(x).all():raise ValueError('nonfinite target')
  return {'x_0':torch.from_numpy(x),'cond':torch.from_numpy(c.copy()).permute(2,0,1)}
 def visualize_sample(self,x_0):
  return x_0[:,:3,:,:,x_0.shape[-1]//2]

def fresh_output(path):
 p=Path(path)
 if p.exists():raise FileExistsError('pilot output already exists; no overwrite/resume by this runner')
 p.mkdir(parents=True)
 return p


def checked_phase_steps(requested,approved):
 if type(requested) is not int or requested not in (50,100):raise ValueError('phase must be50 or100')
 if type(approved) is not int or approved not in (100,200) or requested>approved:raise ValueError('phase exceeds approved budget')
 return requested
