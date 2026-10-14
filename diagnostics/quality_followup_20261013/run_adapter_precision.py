import os,sys,time,json,hashlib,traceback,random
from pathlib import Path
ROOT=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers,LoRALinear
from scripts.utils.inference_contract import TARGETS,HashCache,checked_load,validate_lora,write_manifest
OUT=ROOT/'diagnostics/quality_followup_20261013/adapter_precision';OUT.mkdir(parents=True,exist_ok=False)
H=HashCache();M={'schema':'trellis.adapter-precision.v1','status':'running','training_approved':False,'seed':42,'gpu_physical':6,'no_ss_sampling':True,'no_texture_sampling':True,'formal_acceptance_eligible':False,'results':[]}
def save():write_manifest(OUT/'manifest.json',M)
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def th(t):return ah(t.detach().cpu().float().numpy())
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
original_forward=LoRALinear.forward
try:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
 M['script']=H.record(__file__);M['original_source']=H.record(ROOT/'diagnostics/remediation_20261013/run_shape_flow_compatibility.py');M['lora_source']=H.record(ROOT/'trellis2/modules/lora.py');save()
 p=Trellis2ImageTo3DPipeline.from_pretrained(str(ROOT/'checkpoints'));p.cuda();p.ss_swap_yz=False;p.ss_canonicalize=False
 M['base_sources']={}
 for n,model in p.models.items():
  s=getattr(model,'_checkpoint_source',None)
  if s:M['base_sources'][n]={'config':H.record(s['config_file']),'weights':H.record(s['model_file'])}
 M['shape_sampler']=p.shape_slat_sampler_params;M['shape_normalization']=p.shape_slat_normalization
 ckpt=ROOT/'results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt';M['fixed_checkpoint']=H.record(ckpt)
 model=p.models['shape_slat_flow_model_1024'];names=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.)
 state=torch.load(ckpt,map_location='cpu',weights_only=True);state.pop('rope_phases',None);validate_lora(model,names,state,8,8.,0.);checked_load(model,state,allow_rope=True);del state
 params=list(model.named_parameters());initial={n:(id(x),str(x.dtype),th(x)) for n,x in params};assert sum('.lora_down.' in n or '.lora_up.' in n for n,x in params)==300
 M['rank']=8;M['alpha']=8;M['adapter_count']=150;M['eval']=not model.training
 def check_params():
  for n,x in params:
   assert id(x)==initial[n][0] and str(x.dtype)==initial[n][1] and th(x)==initial[n][2],n
   if '.lora_down.' in n or '.lora_up.' in n:assert x.dtype==torch.float32
 def bf16_branch(self,x):
  import torch.nn.functional as F
  base=F.linear(x.to(dtype=self.weight.dtype),self.weight,self.bias)
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   update=self.lora_up(self.dropout(self.lora_down(x.float())))
  return base+update.to(dtype=base.dtype)*self.scale
 sampler=p.shape_slat_sampler.sample;current={}
 def capture(m,noise,*args,**kwargs):
  current['noise_sha256']=th(noise.feats);current['noise_coords_sha256']=ah(noise.coords.cpu().numpy());return sampler(m,noise,*args,**kwargs)
 p.shape_slat_sampler.sample=capture
 for tag in ['normal','adapter_autocast_bf16']:
  LoRALinear.forward=original_forward if tag=='normal' else bf16_branch
  check_params();assert not torch.is_autocast_enabled()
  for obj,view in [('0086bbab37559442',0),('0fab4af3f9a73bdd',6)]:
   print('CASE_START',tag,obj,flush=True);d=OUT/f'{obj}_{tag}';d.mkdir();r={'object':obj,'tag':tag,'outputs':{},'status':'running'};M['results'].append(r);save()
   sp=ROOT/'diagnostics/remediation_20261013/common_shape_pilot'/obj/'ss_support64_from_shape_encoder.npz';c=np.load(sp,allow_pickle=False)['coords'].astype(np.int32);coords=torch.from_numpy(np.column_stack([np.zeros(len(c),np.int32),c])).cuda();r['support']=H.record(sp);r['coords_sha256']=ah(coords.cpu().numpy());r['token_count']=len(c)
   images=sorted((ROOT/'training_data/multiview_1186/renders_cond_t'/obj).glob(f'{view:03d}_*.png'));assert len(images)==1;r['image']=H.record(images[0])
   with Image.open(images[0]) as im:pre=p.preprocess_image(im.convert('RGBA'))
   pre.save(d/'condition_black.png');r['outputs']['condition']=H.record(d/'condition_black.png');seed()
   with torch.no_grad():cond512=p.get_cond([pre],512);cond1024=p.get_cond([pre],1024);sscond=p.get_ss_cond([pre])
   r['cond512_sha256']=th(cond512['cond']);r['cond1024_sha256']=th(cond1024['cond']);r['sscond_sha256']=th(sscond['cond']);r['cond_dtype']=str(cond1024['cond'].dtype);assert cond1024['cond'].dtype==torch.float32;del cond512,sscond
   r['cpu_rng_before_noise_sha256']=ah(torch.get_rng_state().numpy());current.clear();t=time.perf_counter()
   with torch.no_grad():latent=p.sample_shape_slat(cond1024,model,coords)
   torch.cuda.synchronize();r['flow_seconds']=time.perf_counter()-t;r.update(current);assert torch.isfinite(latent.feats).all()
   np.savez_compressed(d/'sampled_shape_mean.npz',coords=latent.coords.cpu().numpy(),feats=latent.feats.cpu().numpy());r['outputs']['latent']=H.record(d/'sampled_shape_mean.npz');del cond1024
   with torch.no_grad():meshes,subs=p.decode_shape_slat(latent,1024)
   mesh=meshes[0];v=mesh.vertices.cpu().float().numpy();f=mesh.faces.cpu().numpy();assert np.isfinite(v).all() and len(f)>0
   np.savez_compressed(d/'raw_geometry.npz',vertices=v,faces=f);r['outputs']['raw_npz']=H.record(d/'raw_geometry.npz')
   import trimesh
   trimesh.Trimesh(v,f,process=False).export(d/'raw_geometry.ply');vf=np.column_stack([v[:,0],v[:,2],-v[:,1]]);trimesh.Trimesh(vf,f,process=False).export(d/'formal_geometry_only.ply')
   for key,filename in [('raw_ply','raw_geometry.ply'),('formal_ply','formal_geometry_only.ply')]:r['outputs'][key]=H.record(d/filename)
   r['status']='success';r['adapter_dtype']='torch.float32';r['vertices']=len(v);r['faces']=len(f);save();del latent,meshes,subs,mesh,coords;torch.cuda.empty_cache()
 for obj in ['0086bbab37559442','0fab4af3f9a73bdd']:
  pair=[x for x in M['results'] if x['object']==obj];fields=['coords_sha256','cond512_sha256','cond1024_sha256','sscond_sha256','cpu_rng_before_noise_sha256','noise_sha256','noise_coords_sha256'];checks={k:pair[0][k]==pair[1][k] for k in fields};assert all(checks.values());M.setdefault('single_variable_checks',{})[obj]=checks
 check_params();M['parameters_identity_dtype_hash_unchanged']=True;M['status']='success';save();print('ADAPTER_PRECISION_COMPLETE',flush=True)
except Exception as e:
 M.update(status='failed',error=str(e),traceback=traceback.format_exc());save();traceback.print_exc();sys.exit(1)
finally:LoRALinear.forward=original_forward
