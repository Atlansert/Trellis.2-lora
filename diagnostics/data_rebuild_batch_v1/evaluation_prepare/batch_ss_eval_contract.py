"""CPU-only SS evaluation contract; never approves training or runs models."""
import hashlib,json
from pathlib import Path
import numpy as np
TRAIN={'0086bbab37559442','008ae427b2606cc8','015de7a65b71fbc2','1208e4cc0c30448a'}
HOLDOUT={'0390f705ac242a39','1f7cc1ae1cec73f5'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def artifact(r):
 p=Path(r['path'])
 if not p.is_absolute() or '..' in p.parts or not p.is_file():raise ValueError('absolute existing nontraversing artifact required')
 if sha(p)!=r.get('sha256'):raise ValueError('artifact checksum changed')
 return p
def validate_inputs(render_gate,encoded_manifest,pilot_manifest,checkpoint_dir=None,step=50,approval_path=None):
 gate=json.loads(Path(render_gate).read_text());encoded=json.loads(Path(encoded_manifest).read_text());pilot=json.loads(Path(pilot_manifest).read_text())
 if gate.get('schema')!='trellis.batch-render-validation.v1' or gate.get('passed') is not True or gate.get('approved_for_encoding') is not True:raise ValueError('render gate not passed')
 if encoded.get('schema')!='trellis.render-batch-encoded.v1':raise ValueError('unknown encoded frame manifest')
 if pilot.get('schema')!='trellis.six-render-ss-pilot.v1' or pilot.get('target_frame')!='proper_raw_depth_length_up':raise ValueError('wrong target frame')
 if set(pilot['train_objects'])!=TRAIN or set(pilot['internal_holdout_objects'])!=HOLDOUT or TRAIN&HOLDOUT:raise ValueError('train/holdout scope changed')
 entries=pilot['objects'];ids=[x['object_id'] for x in entries]
 if len(ids)!=6 or set(ids)!=TRAIN|HOLDOUT:raise ValueError('six asset scope changed')
 if {x['object_id'] for x in encoded['assets']}!=set(ids):raise ValueError('encoded scope changed')
 if set(gate['object_ids'])!=set(ids):raise ValueError('render scope changed')
 ep=Path(encoded_manifest);commit=json.loads((ep.parent/'COMPLETE.json').read_text())
 if commit['manifest_sha256']!=sha(ep):raise ValueError('encoded commit changed')
 if encoded['batch_validation_sha256']!=sha(render_gate):raise ValueError('render gate identity changed')
 dependencies=pilot.get('dependencies',{})
 for key,path in [('encoded_manifest',encoded_manifest),('render_batch_validation',render_gate)]:
  bound=dependencies.get(key)
  if bound is None or artifact(bound).resolve()!=Path(path).resolve():raise ValueError('pilot bound input identity mismatch')
 for key in ['base','recipe','source_batch_review']:artifact(dependencies[key])
 encoded_by={x['object_id']:x for x in encoded['assets']}
 from PIL import Image
 for o in entries:
  asset=encoded_by[o['object_id']]
  if o['target']['sha256']!=asset['stages']['ss']['files']['ss_mean.npz']:raise ValueError('pilot target not encoded target')
  split='train' if o['object_id'] in TRAIN else 'internal_holdout'
  if o.get('split') not in (split,'holdout' if split=='internal_holdout' else split):raise ValueError('split inconsistent')
  if o.get('view_index')!=0:raise ValueError('only audited camera0 condition allowed')
  for key in ['render_rgba','render_manifest','canonical_geometry','target','condition','source_review']:artifact(o[key])
  with Image.open(o['condition']['path']) as im:
   if im.mode!='RGB' or im.size!=(512,512):raise ValueError('cached RGB512 condition required')
  with np.load(o['target']['path'],allow_pickle=False) as z:mean=z['mean']
  if mean.shape!=(8,16,16,16) or mean.dtype!=np.float32 or not np.isfinite(mean).all():raise ValueError('invalid SS mean')
  support=ep.parent/o['object_id']/'ss_support64.npz'
  if sha(support)!=asset['stages']['encode']['files']['ss_support64.npz']:raise ValueError('target support changed')
  with np.load(support,allow_pickle=False) as z:coords=z['coords']
  occupancy(coords)
 if checkpoint_dir is not None:
  # Only imports CPU torch and checkpoint validation; no models/CUDA creation.
  from trellis2.utils.checkpoint_contract import validate_checkpoint
  d=Path(checkpoint_dir);m=validate_checkpoint(d,step,verify_hashes=True)
  required={f'denoiser_step{step:07d}.pt',f'denoiser_ema0.9999_step{step:07d}.pt',f'misc_step{step:07d}.pt'}
  if {x['name'] for x in m['files']}!=required:raise ValueError('checkpoint topology unexpected')
  runtime_path=d.parent/'pilot_run_contract.json'
  if not runtime_path.exists():raise ValueError('runtime pilot identity missing')
  runtime=json.loads(runtime_path.read_text())
  # Trainer writer contract inspected by parent; tolerate only explicit keys,
  # never infer identity from directory or step filename.
  digest=runtime.get('manifest_sha256')
  if digest!=sha(pilot_manifest):raise ValueError('trained pilot manifest identity mismatch')
  if runtime.get('phase_steps',runtime.get('steps'))!=step:raise ValueError('checkpoint phase mismatch')
  rt_train=runtime.get('train_objects',runtime.get('scope'))
  if rt_train is None or set(rt_train)!=TRAIN:raise ValueError('trained object scope mismatch')
  if set(runtime.get('internal_holdout_objects',[]))!=HOLDOUT:raise ValueError('trained holdout identity mismatch')
  if runtime.get('seed')!=42 or runtime.get('trainable_parameters')!=4669440 or runtime.get('production_approved') is not False:raise ValueError('unknown pilot recipe/runtime identity')
  if runtime.get('base')!=dependencies['base']:raise ValueError('trained base record mismatch')
  approval_hash=runtime.get('approval_sha256')
  if not isinstance(approval_hash,str) or len(approval_hash)!=64:raise ValueError('runtime approval identity missing')
  if approval_path is None or sha(approval_path)!=approval_hash:raise ValueError('actual approval artifact missing/changed')
  approval=json.loads(Path(approval_path).read_text())
  if approval.get('approved') is not True or approval.get('manifest_sha256')!=sha(pilot_manifest):raise ValueError('approval scope identity invalid')
 return pilot,encoded

def occupancy(coords):
 c=np.asarray(coords)
 if c.ndim!=2 or c.shape[1] not in [3,4]:raise ValueError('Nx3 or batch0 Nx4 coordinates required')
 if c.shape[1]==4:
  if not np.all(c[:,0]==0):raise ValueError('only sample batch0 supported')
  c=c[:,1:]
 if not np.isfinite(c).all() or not np.equal(c,np.floor(c)).all() or (c<0).any() or (c>=64).any():raise ValueError('integer finite grid64 coordinates required')
 v=np.zeros((64,64,64),bool)
 if len(c):v[tuple(c.astype(int).T)]=True
 return v

def metrics(coords,gt=None):
 v=occupancy(coords);c=np.argwhere(v).astype(float)
 result={'count':len(c),'frame':'depthX_lengthY_upZ','alignment':'none','fidelity_status':'unverified'}
 if not len(c):return dict(result,spans_depth_length_up=[0,0,0],longest_index=None,axis_status='empty')
 spans=np.ptp(c,axis=0)+1;center=c.mean(0)
 result.update(spans_depth_length_up=spans.tolist(),centroid=center.tolist(),longest_index=int(spans.argmax()),cross_section_depth_over_up=float(spans[0]/spans[2]))
 if len(c)>1:
  cov=np.cov(c,rowvar=False);vals,vec=np.linalg.eigh(cov);axis=vec[:,-1];result.update(covariance=cov.tolist(),pca_major_axis=axis.tolist(),length_offangle_degrees=float(np.degrees(np.arccos(np.clip(abs(axis[1]),0,1)))),centroid_band_high_low_ratio=float((c[:,2]>32).sum()/max((c[:,2]<=32).sum(),1)))
 if gt is not None:
  target=occupancy(gt);inter=(v&target).sum();union=(v|target).sum();result['target']={'iou':float(inter/max(union,1)),'precision':float(inter/max(v.sum(),1)),'recall':float(inter/max(target.sum(),1)),'gt_count':int(target.sum())}
 else:result['no_gt']='external numbered image has no confirmed asset identity; no fidelity IoU'
 return result

def summary(cases):
 rows=[]
 for split in ['train','internal_holdout']:
  base=[x['target']['iou'] for x in cases if x['split']==split and x['tag']=='base'];live=[x['target']['iou'] for x in cases if x['split']==split and x['tag']=='live']
  if len(base)!=len(live) or not base:raise ValueError('comparison incomplete')
  rows.append({'split':split,'objects':len(base),'base_mean_iou':float(np.mean(base)),'live_mean_iou':float(np.mean(live)),'base_min_iou':float(min(base)),'live_min_iou':float(min(live)),'mean_change':float(np.mean(live)-np.mean(base)),'not_production_acceptance':True})
 return rows

def fresh_output(path):
 p=Path(path)
 if p.exists():raise FileExistsError('existing evaluation evidence must not be overwritten')
 p.mkdir(parents=True);return p

def atomic_json(path,data):
 p=Path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');tmp.replace(p)
