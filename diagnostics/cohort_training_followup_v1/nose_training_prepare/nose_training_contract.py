import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
TRAIN=['0086bbab37559442','008ae427b2606cc8','015de7a65b71fbc2','1208e4cc0c30448a']
HOLDOUT=['0390f705ac242a39','1f7cc1ae1cec73f5']
CHANGED={'1208e4cc0c30448a','1f7cc1ae1cec73f5'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def artifact(r):
 p=Path(r['path'])
 if not p.is_absolute() or '..' in p.parts or not p.is_file() or sha(p)!=r['sha256']:raise ValueError('artifact changed/unsafe')
 return p
def finite_mean(p):
 with np.load(p,allow_pickle=False) as z:a=z['mean'].copy()
 if a.shape!=(8,16,16,16) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('finite FP32 mean required')
 return a
def rotation(R,changed):
 R=np.asarray(R);E=np.diag([-1,-1,1]) if changed else np.eye(3)
 if R.shape!=(3,3) or not np.allclose(R,E,atol=1e-12) or abs(np.linalg.det(R)-1)>1e-12:raise ValueError('proper declared rotation required')
def fresh(p):
 p=Path(p)
 if p.exists():raise FileExistsError('preserve existing output')
 p.mkdir(parents=True);return p
def approval_check(a,old,new,candidate,baseline,base_sha):
 if a.get('schema')!='trellis.nose-canonical-target-trial-approval.v1' or a.get('approved') is not True or a.get('diagnostic_only') is not True or a.get('production_approved') is not False or a.get('max_steps')!=100:raise ValueError('explicit100 diagnostic approval required')
 if a.get('train_objects')!=TRAIN or a.get('internal_holdout_objects')!=HOLDOUT or a.get('condition_view')!=0 or a.get('base_sha256')!=base_sha:raise ValueError('approval scope drift')
 for k,p in [('old_manifest_sha256',old),('new_target_manifest_sha256',new),('candidate_manifest_sha256',candidate),('old_baseline100_run_contract_sha256',baseline)]:
  if a.get(k)!=sha(p):raise ValueError('approval hash mismatch')
 gate=json.loads(artifact(a['target_validation']).read_text())
 if gate.get('schema')!='trellis.nose-target-independent-validation.v1' or gate.get('approved_for_scoped_target_trial') is not True or gate.get('production_approved') is not False or gate.get('conditions_changed') is not False:raise ValueError('scoped newtarget gate required')
 if gate.get('passed') is not True or gate.get('encoded_manifest_sha256')!=sha(new) or set(gate.get('objects',gate.get('scope',[r['object_id'] for r in gate.get('rows',[])])))!=set(TRAIN+HOLDOUT):raise ValueError('newtarget validation missing/wrong scope')
def validate(old_path,new_path,candidate_path,baseline_path,approval=None):
 old_path=Path(old_path);new_path=Path(new_path);candidate_path=Path(candidate_path);baseline_path=Path(baseline_path)
 old=json.loads(old_path.read_text());new=json.loads(new_path.read_text());cand=json.loads(candidate_path.read_text());baseline=json.loads(baseline_path.read_text())
 for p in [old_path,new_path,candidate_path]:
  if json.loads(p.with_name('COMPLETE.json').read_text())['manifest_sha256']!=sha(p):raise ValueError('manifest completion mismatch')
 if old['train_objects']!=TRAIN or old['internal_holdout_objects']!=HOLDOUT or baseline['train_objects']!=TRAIN or baseline['internal_holdout_objects']!=HOLDOUT or baseline['phase_steps']!=100:raise ValueError('split/baseline100 mismatch')
 if new.get('schema')!='trellis.nose-target-encoded.v1' or new.get('training_approved') is not False or new.get('conditions_changed') is not False or new.get('actual_reencoded_assets')!=2 or new.get('identity_targets_reused')!=4:raise ValueError('encoded contract invalid')
 if new['candidate_manifest_sha256']!=sha(candidate_path):raise ValueError('candidate changed')
 rows=new['records'];by={r['object_id']:r for r in rows};original={r['object_id']:r for r in old['objects']};candidates={r['object_id']:r for r in cand['records']}
 if len(rows)!=6 or set(by)!=set(TRAIN+HOLDOUT) or set(candidates)!=set(by):raise ValueError('exact6 scope required')
 for oid,r in by.items():
  o=original[oid];c=candidates[oid];changed=oid in CHANGED;rotation(c['matrix_row_oldraw_to_canonical'],changed)
  if r['changed_geometry'] is not changed or r['split']!=('train' if oid in TRAIN else 'internal_holdout'):raise ValueError('target/split drift')
  cp=artifact(o['condition']);npth=artifact(r['condition_unchanged'])
  if cp.resolve()!=npth.resolve() or o['condition']['sha256']!=r['condition_unchanged']['sha256']:raise ValueError('condition drift')
  with Image.open(cp) as im:
   if im.mode!='RGB' or im.size!=(512,512):raise ValueError('RGB512 required')
  gp=artifact({'path':r['geometry_path'],'sha256':r['geometry_sha256']});src=artifact(c['source_geometry'])
  with np.load(gp,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
  with np.load(src,allow_pickle=False) as z:sv=z['vertices'];sf=z['faces']
  if not np.isfinite(v).all() or np.argmax(np.ptp(v,axis=0))!=1 or not np.array_equal(f,sf) or not np.allclose(v,sv@np.asarray(c['matrix_row_oldraw_to_canonical']),atol=1e-7,rtol=0):raise ValueError('wrong geometry/rotation/frame')
  for stage in ['encode','decode','ss']:
   st=r['stages'][stage];record=st['record'];folder=Path(st['reference_folder'])
   if st['reused_identity'] is not (not changed) or record['input_geometry_sha256']!=r['geometry_sha256']:raise ValueError('stage identity mismatch')
   for name,h in record['files'].items():
    if Path(name).name!=name or sha(folder/name)!=h:raise ValueError('stage artifact changed')
  if r['stages']['ss']['record']['result']['support_iou']!=1:raise ValueError('SS roundtrip failed')
  finite_mean(artifact(r['ss_mean_reference']));artifact(r['shape_mean_reference'])
 for dep in old['dependencies'].values():artifact(dep)
 if baseline['base']['sha256']!=old['dependencies']['base']['sha256'] or baseline['seed']!=old['seed']:raise ValueError('baseline init differs')
 if approval is not None:approval_check(json.loads(Path(approval).read_text()),old_path,new_path,candidate_path,baseline_path,old['dependencies']['base']['sha256'])
 return old,by,baseline
class Dataset:
 value_range=(0,1);normalization=None;skip_snapshot=True
 def __init__(self,old,new,candidate,baseline,approval):
  self.m,self.targets,self.baseline=validate(old,new,candidate,baseline,approval);self.original={r['object_id']:r for r in self.m['objects']};self.objects=TRAIN.copy()
 def __len__(self):return 4
 def __getitem__(self,i):
  import torch
  oid=self.objects[i];cp=artifact(self.original[oid]['condition']);target=artifact(self.targets[oid]['ss_mean_reference'])
  with Image.open(cp) as im:c=np.asarray(im,dtype=np.float32)/255
  return {'x_0':torch.from_numpy(finite_mean(target)),'cond':torch.from_numpy(c.copy()).permute(2,0,1)}
 def visualize_sample(self,x):
  if isinstance(x,dict):x=x['x_0']
  return x[:,:3,:,:,x.shape[-1]//2]
