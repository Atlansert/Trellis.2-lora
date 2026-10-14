"""Prepared evaluation ONLY. --execute requires actual committed live checkpoint.
No training, shape/texture sampling, axis corrections or production acceptance.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
ROOT=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from batch_ss_eval_contract import validate_inputs,metrics,summary,sha,fresh_output,atomic_json
p=argparse.ArgumentParser();p.add_argument('--render-gate',required=True);p.add_argument('--encoded-manifest',required=True);p.add_argument('--pilot-manifest',required=True);p.add_argument('--checkpoint-dir');p.add_argument('--approval');p.add_argument('--validate-inputs-only',action='store_true');p.add_argument('--step',type=int,choices=[50,100],default=50);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
if a.validate_inputs_only and a.execute:raise ValueError('input validation cannot execute models')
if not a.validate_inputs_only and a.checkpoint_dir is None:raise ValueError('actual committed checkpoint directory required')
pilot,encoded=validate_inputs(a.render_gate,a.encoded_manifest,a.pilot_manifest,None if a.validate_inputs_only else a.checkpoint_dir,a.step,a.approval)
if not a.execute:print(json.dumps({'input_contract_passed':True,'checkpoint_commit_passed':not a.validate_inputs_only,'models_loaded':False,'training_started':False}));sys.exit(0)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise ValueError('evaluation requires single GPU6 only')
import numpy as np,torch
from PIL import Image,ImageDraw
from safetensors import safe_open
from scripts.utils.inference_contract import checked_load,validate_lora,TARGETS,HashCache
from trellis2.modules.lora import apply_lora_to_linear_layers
from trellis2.pipelines import Trellis2ImageTo3DPipeline
out=fresh_output(a.output);hc=HashCache();cp=Path(a.checkpoint_dir)/f'denoiser_step{a.step:07d}.pt'
report={'schema':'trellis.six-asset-SS-evaluation.v1','status':'checking','ssproxy_not_acceptance':True,'training_continued':False,'production_approved':False,'seed':42,'step':a.step,'cases':[],'sources':{k:hc.record(v) for k,v in [('render_gate',a.render_gate),('encoded_manifest',a.encoded_manifest),('pilot_manifest',a.pilot_manifest),('checkpoint',cp),('script',__file__),('contract',Path(__file__).with_name('batch_ss_eval_contract.py'))]}}
def save():atomic_json(out/'metrics.json',report)
def tensorhash(t):return hashlib.sha256(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()
try:
 base=ROOT/'checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors';state=torch.load(cp,map_location='cpu',weights_only=True,mmap=True);ad={k:v for k,v in state.items() if '.lora_' in k};changed=[];nonfinite=[];missing=[]
 with safe_open(str(base),framework='pt',device='cpu') as f:
  keys=set(f.keys())
  for k,v in state.items():
   if not torch.isfinite(v).all():nonfinite.append(k)
   if k not in ad and k!='rope_phases':
    if k not in keys:missing.append(k)
    elif not torch.equal(v,f.get_tensor(k).to(v.dtype)):changed.append(k)
 report['integrity']={'adapter_tensors':len(ad),'adapter_fp32':all(v.dtype==torch.float32 for v in ad.values()),'nonfinite':nonfinite,'frozen_changed':changed,'unexpected_frozen':missing};assert len(ad)==300 and report['integrity']['adapter_fp32'] and not changed and not nonfinite and not missing;del ad,state;save()
 t=time.monotonic();pipe=Trellis2ImageTo3DPipeline.from_pretrained(str(ROOT/'checkpoints'));pipe.cuda();assert not pipe.ss_canonicalize and not pipe.ss_swap_yz;report['load_seconds']=time.monotonic()-t;report['sampler']=pipe.sparse_structure_sampler_params
 report['actual_base_sources']={}
 for key,model in pipe.models.items():
  source=getattr(model,'_checkpoint_source',None)
  if source:report['actual_base_sources'][key]={'config':hc.record(source['config_file']),'weights':hc.record(source['model_file'])}
 inputs=[]
 for o in pilot['objects']:
  support=Path(a.encoded_manifest).parent/o['object_id']/'ss_support64.npz'
  with np.load(support,allow_pickle=False) as z:gt=z['coords']
  inputs.append({'id':o['object_id'],'split':'train' if o['object_id'] in pilot['train_objects'] else 'internal_holdout','path':o['condition']['path'],'preprocess':False,'gt':gt,'support_record':hc.record(support)})
 for n in ['001','002','003','004']:inputs.append({'id':'external'+n,'split':'external','path':str(ROOT/'data_test'/f'{n}.png'),'preprocess':True,'gt':None})
 current={};sampler=pipe.sparse_structure_sampler.sample
 def capture(model,noise,*args,**kwargs):
  current['noise_sha256']=tensorhash(noise);return sampler(model,noise,*args,**kwargs)
 pipe.sparse_structure_sampler.sample=capture
 for tag in ['base','live']:
  if tag=='live':
   model=pipe.models['sparse_structure_flow_model'];replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.);assert len(replaced)==150
   state=torch.load(cp,map_location='cpu',weights_only=True);state.pop('rope_phases',None);validate_lora(model,replaced,state,8,8.,0.);checked_load(model,state);del state
  for inp in inputs:
   name=inp['id'];t=time.monotonic()
   with Image.open(inp['path']) as im:image=im.convert('RGBA' if inp['preprocess'] else 'RGB')
   image=pipe.preprocess_image(image) if inp['preprocess'] else image
   # Cache RGB512 is already production cropped, never crop it twice.
   image.save(out/f'{tag}_{name}_condition.png');torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   with torch.inference_mode():
    c512=pipe.get_cond([image],512);c1024=pipe.get_cond([image],1024);css=pipe.get_ss_cond([image]);condition=tensorhash(css['cond']);rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest();current.clear();coords=pipe.sample_sparse_structure(css,64,1,{}).cpu().numpy()
   artifact=out/f'{tag}_{name}.npz';np.savez_compressed(artifact,coords=coords)
   row={'tag':tag,'input':name,'split':inp['split'],'input_artifact':hc.record(inp['path']),'cached_condition_not_recropped':not inp['preprocess'],'condition_sha256':condition,'cpu_rng_before_sampling':rng,**current,'seconds':time.monotonic()-t,'coords_artifact':hc.record(artifact),**metrics(coords,inp['gt'])}
   if inp['gt'] is not None:row['target_support']=inp['support_record']
   report['cases'].append(row);save();print('CASE_DONE',tag,name,json.dumps({'count':row['count'],'target':row.get('target'),'span':row['spans_depth_length_up']}),flush=True);del c512,c1024,css
 for inp in inputs:
  rows=[x for x in report['cases'] if x['input']==inp['id']];assert len(rows)==2
  for key in ['condition_sha256','cpu_rng_before_sampling','noise_sha256']:assert rows[0][key]==rows[1][key],(inp['id'],key)
 report['split_summary']=summary(report['cases']);report['same_condition_rng_noise_verified']=True
 # Fixed raw-camera projections: horizontal lengthY, upZ. No alignment.
 canvas=Image.new('RGB',(900,6*130),(20,22,26));draw=ImageDraw.Draw(canvas)
 for i,inp in enumerate(inputs[:6]):
  y=i*130;draw.text((5,y+3),inp['id']+' '+inp['split'],fill='white')
  for col,(label,c) in enumerate([('target',inp['gt']),('base',np.load(out/f"base_{inp['id']}.npz")['coords'][:,1:]),('live',np.load(out/f"live_{inp['id']}.npz")['coords'][:,1:])]):
   draw.text((col*300+5,y+20),label,fill='white')
   for pt in c:draw.rectangle((col*300+10+int(pt[1]*4),y+45+int((63-pt[2])*1),col*300+14+int(pt[1]*4),y+46+int((63-pt[2])*1)),fill=(95,185,225))
 canvas.save(out/'six_fixed_SS_projections.png');report.update(status='success',EMA_not_primary='50-stepEMA retains about99.5percent initialization; live used',formal_GLb_acceptance=False);save();atomic_json(out/'COMPLETE.json',{'metrics_sha256':sha(out/'metrics.json'),'training_continued':False,'production_approved':False});print('SIX_SS_EVALUATION_COMPLETE',flush=True)
except Exception as exc:
 report.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise
