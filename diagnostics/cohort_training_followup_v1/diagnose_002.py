import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path('/data/lzx/TRELLIS.2');os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'diagnostics/data_rebuild_batch_v1/evaluation_prepare'))
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
import torch,numpy as np
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers,LoRALinear
from scripts.utils.inference_contract import checked_load,validate_lora,TARGETS,HashCache,write_manifest
from batch_ss_eval_contract import metrics
out=ROOT/'diagnostics/cohort_training_followup_v1/axis002_v1';out.mkdir(exist_ok=False);H=HashCache();report={'schema':'trellis.axis002-diagnostic.v1','status':'running','no_training':True,'no_axis_correction':True,'no_shape_texture':True,'production_approved':False,'cases':[]}
def save():write_manifest(out/'manifest.json',report)
def th(t):return hashlib.sha256(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()
save();pipe=Trellis2ImageTo3DPipeline.from_pretrained(str(ROOT/'checkpoints'));pipe.cuda();assert not pipe.ss_canonicalize and not pipe.ss_swap_yz
image=pipe.preprocess_image(Image.open(ROOT/'data_test/002.png').convert('RGBA'));image.save(out/'condition.png');report['image']=H.record(ROOT/'data_test/002.png');report['condition']=H.record(out/'condition.png');report['sampler']=pipe.sparse_structure_sampler_params
checkpoints={'cohort50':ROOT/'diagnostics/data_rebuild_batch_v1/ss_short_train4_holdout2_step50/ckpts/denoiser_step0000050.pt','axis10_ema20k':ROOT/'results/multiview_1186_ss_flow_lora_64_canon_axis10/ckpts/denoiser_ema0.9999_step0020000.pt'}
model=pipe.models['sparse_structure_flow_model'];current={};original_sample=pipe.sparse_structure_sampler.sample
def sample_capture(model,noise,*args,**kw):
 current['noise_sha256']=th(noise);return original_sample(model,noise,*args,**kw)
pipe.sparse_structure_sampler.sample=sample_capture
for tag in ['base','cohort50','axis10_ema20k']:
 if tag!='base':
  if tag=='cohort50':replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.);assert len(replaced)==150
  else:replaced=[n for n,m in model.named_modules() if isinstance(m,LoRALinear)];assert len(replaced)==150
  state=torch.load(checkpoints[tag],map_location='cpu',weights_only=True);state.pop('rope_phases',None);validate_lora(model,replaced,state,8,8.,0.);checked_load(model,state);del state
  report.setdefault('checkpoints',{})[tag]=H.record(checkpoints[tag])
 for seed in [42,0,1]:
  started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);current.clear()
  with torch.inference_mode():
   cond512=pipe.get_cond([image],512);cond1024=pipe.get_cond([image],1024);cond=pipe.get_ss_cond([image]);condition_hash=th(cond['cond']);rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest();coords=pipe.sample_sparse_structure(cond,64,1,{}).cpu().numpy()
  path=out/f'{tag}_seed{seed}.npz';np.savez_compressed(path,coords=coords);row={'tag':tag,'seed':seed,'condition_sha256':condition_hash,'cpu_rng_before_sampling':rng,**current,'coords':H.record(path),'seconds':time.monotonic()-started,**metrics(coords,None)};report['cases'].append(row);save();print('RESULT',json.dumps({k:row[k] for k in ['tag','seed','count','spans_depth_length_up','length_offangle_degrees']}),flush=True);del cond512,cond1024,cond
for seed in [42,0,1]:
 rows=[r for r in report['cases'] if r['seed']==seed]
 for key in ['condition_sha256','cpu_rng_before_sampling','noise_sha256']:assert len({r[key] for r in rows})==1,(seed,key)
report.update(status='success',all_condition_and_noise_controls_verified=True,interpretation_limit='axis10 older target frame differs in up/handedness; length success alone not production quality; otherseeds reference not acceptance')
save();write_manifest(out/'COMPLETE.json',{'manifest_sha256':H.record(out/'manifest.json')['sha256'],'production_approved':False});print('AXIS002_DIAGNOSTIC_COMPLETE',flush=True)
