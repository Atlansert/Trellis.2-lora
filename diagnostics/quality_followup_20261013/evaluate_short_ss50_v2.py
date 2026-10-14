import os,sys,json,time,hashlib
from pathlib import Path
R=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(R));os.chdir(R)
import numpy as np,torch
from PIL import Image,ImageDraw
from safetensors import safe_open
from trellis2 import models
from scripts.utils.inference_contract import checked_load,validate_lora,TARGETS
from trellis2.modules.lora import apply_lora_to_linear_layers
from trellis2.pipelines import Trellis2ImageTo3DPipeline
B=R/'diagnostics/quality_followup_20261013';O=B/'short_ss_50_evaluation_v2';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def eq(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
plan=json.loads((B/'short_ss_pilot_plan/manifest.json').read_text());obj=plan['objects'][0];integ={};miscs=[]
base=R/'checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors'
with safe_open(str(base),framework='pt',device='cpu') as f:
 for arm in ['legacy','proper']:
  ck=B/f'short_ss_{arm}_50/ckpts/denoiser_step0000050.pt';s=torch.load(ck,map_location='cpu',weights_only=True,mmap=True);ad={k:v for k,v in s.items() if '.lora_' in k};changed=[];nonfinite=[]
  for k,v in s.items():
   if not torch.isfinite(v).all():nonfinite.append(k)
   if k not in ad and k!='rope_phases' and not torch.equal(v,f.get_tensor(k).to(v.dtype)):changed.append(k)
  misc=torch.load(ck.parent/'misc_step0000050.pt',map_location='cpu',weights_only=False);miscs.append(misc)
  integ[arm]={'tensors':len(s),'adapter_tensors':len(ad),'adapter_all_fp32':all(v.dtype==torch.float32 for v in ad.values()),'frozen_changed':changed,'nonfinite':nonfinite,'misc_step':misc['step'],'adam_steps':sorted({int(x['step']) for x in misc['optimizer']['state'].values()}),'adam_moment_dtypes':sorted({str(x[k].dtype) for x in misc['optimizer']['state'].values() for k in ['exp_avg','exp_avg_sq']}),'checkpoint_sha256':sha(ck)}
  assert not changed and not nonfinite and len(ad)==300 and integ[arm]['adapter_all_fp32'];del s,ad
same_rng=eq(miscs[0]['rng_state'],miscs[1]['rng_state']);del miscs
(O/'integrity.json').write_text(json.dumps({'checks':integ,'after_training_rng_equal':same_rng},indent=2));print('INTEGRITY',integ,'RNG_EQUAL',same_rng,flush=True)
p=Trellis2ImageTo3DPipeline.from_pretrained(str(R/'checkpoints'));p.cuda();assert not p.ss_swap_yz and not p.ss_canonicalize
M={'schema':'trellis.short-ss50-evaluation.v1','seed':42,'no_shape_texture_sampling':True,'ssproxy_not_acceptance':True,'training_approved':False,'integrity':integ,'after_training_rng_equal':same_rng,'cases':[],'sampler':p.sparse_structure_sampler_params};targets={}
for arm in ['legacy','proper']:
 decoder=p.models['sparse_structure_decoder'];decoder.convert_to_fp32();decoder.float().cpu().eval()
 z=torch.from_numpy(np.load(obj['targets'][arm]['path'])['mean']).unsqueeze(0)
 with torch.inference_mode():occ=(decoder(z)>0)[0,0].cpu().numpy()
 targets[arm]=occ;np.savez_compressed(O/f'target_{arm}.npz',coords=np.argwhere(occ))
 del z
inputs=[('train0086',Image.open(obj['condition']['path']).convert('RGB'),False),('001',Image.open(R/'data_test/001.png').convert('RGBA'),True),('004',Image.open(R/'data_test/004.png').convert('RGBA'),True)]
def stats(coords):
 c=coords[:,1:].astype(float);span=np.ptp(c,axis=0)+1;cov=np.cov(c,rowvar=False);v,w=np.linalg.eigh(cov);axis=w[:,-1];angle=float(np.degrees(np.arccos(np.clip(abs(axis[1]),0,1))))
 r={'count':len(c),'spans_depth_length_up':span.tolist(),'pca_major_axis':axis.tolist(),'length_offangle_degrees':angle,'covariance':cov.tolist(),'longest_index':int(span.argmax())}
 occ=np.zeros((64,64,64),bool);occ[tuple(c.astype(int).T)]=True
 for tag,gt in targets.items():
  it=(occ&gt).sum();r[tag+'_target']={'iou':float(it/(occ|gt).sum()),'precision':float(it/max(occ.sum(),1)),'recall':float(it/max(gt.sum(),1))}
 return r
for tag in ['base','legacy_live','proper_live','legacy_ema','proper_ema']:
 if tag!='base':
  arm=tag.split('_')[0];suffix='denoiser_step0000050.pt' if tag.endswith('live') else 'denoiser_ema0.9999_step0000050.pt';model=p.models['sparse_structure_flow_model'];replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.)
  if not replaced:replaced=[n for n,m in model.named_modules() if hasattr(m,'lora_down') and hasattr(m,'lora_up')]
  state=torch.load(B/f'short_ss_{arm}_50/ckpts'/suffix,map_location='cpu',weights_only=True);state.pop('rope_phases',None);validate_lora(model,replaced,state,8,8.,0.);checked_load(model,state);del state
 for name,image,pre in inputs:
  t=time.monotonic();im=p.preprocess_image(image) if pre else image.copy();torch.manual_seed(42)
  with torch.inference_mode():
   c512=p.get_cond([im],512);c1024=p.get_cond([im],1024);css=p.get_ss_cond([im]);ch=th(css['cond']);coords=p.sample_sparse_structure(css,64,1,{}).cpu().numpy()
  np.savez_compressed(O/f'{tag}_{name}.npz',coords=coords);r={'tag':tag,'input':name,'condition_sha256':ch,'seconds':time.monotonic()-t,**stats(coords)}
  if name!='train0086':r.pop('legacy_target');r.pop('proper_target');r['no_gt']='heldout no asset identity, no IoU acceptance'
  M['cases'].append(r);print('RESULT',json.dumps(r),flush=True);del c512,c1024,css
  (O/'metrics.json').write_text(json.dumps(M,indent=2))
# quick fixed camera max occupancy projections, no fitting/rotation
canvas=Image.new('RGB',(1000,420),(20,22,25));d=ImageDraw.Draw(canvas)
sets=[('target legacy',np.argwhere(targets['legacy'])),('target proper',np.argwhere(targets['proper']))]+[(tag,np.load(O/f'{tag}_train0086.npz')['coords'][:,1:]) for tag in ['base','legacy_live','proper_live']]
for j,(label,c) in enumerate(sets):
 d.text((j*200+5,5),label,fill='white')
 for row,(h,v) in enumerate([(1,2),(1,0),(0,2)]):
  for pt in c:d.rectangle((j*200+10+int(pt[h]*2.8),row*130+35+int((63-pt[v])*1.7),j*200+12+int(pt[h]*2.8),row*130+37+int((63-pt[v])*1.7)),fill=(100,190,220))
canvas.save(O/'train_fixed_projections.png');(O/'COMPLETE.json').write_text(json.dumps({'metrics_sha256':sha(O/'metrics.json'),'training_continued':False,'no_acceptance_claim':True},indent=2));print('EVALUATION_COMPLETE',flush=True)
