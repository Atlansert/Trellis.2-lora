import os,sys,time,json,hashlib,traceback,random
from pathlib import Path
ROOT=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers
from scripts.utils.inference_contract import TARGETS,HashCache,checked_load,validate_lora,glb_coordinates,write_manifest
OUT=ROOT/'diagnostics/remediation_20261013/shape_flow_compatibility'
OUT.mkdir(exist_ok=False)
H=HashCache();M={'schema':'trellis.shape-flow-compatibility.v1','status':'running','training_approved':False,'seed':42,'gpu_physical':6,'no_ss_sampling':True,'no_texture_sampling':True,'formal_acceptance_eligible':False,'results':[],'started_unix':time.time()}
def save():write_manifest(OUT/'manifest.json',M)
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def tensorhash(t):return ah(t.detach().cpu().float().numpy())
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def stamp(name,start):torch.cuda.synchronize();print(name,'seconds',time.monotonic()-start,flush=True);return time.monotonic()-start
try:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
 M['dependencies']={'torch':torch.__version__,'numpy':np.__version__,'python':sys.version}
 M['sources']={str(p):H.record(p) for p in [Path(__file__),ROOT/'trellis2/pipelines/trellis2_image_to_3d.py',ROOT/'scripts/utils/inference_contract.py',ROOT/'checkpoints/pipeline.json']}
 save();t=time.monotonic();p=Trellis2ImageTo3DPipeline.from_pretrained(str(ROOT/'checkpoints'));p.cuda();p.ss_swap_yz=False;p.ss_canonicalize=False;M['load_seconds']=stamp('LOAD',t)
 M['base_sources']={}
 for n,model in p.models.items():
  s=getattr(model,'_checkpoint_source',None)
  if s:M['base_sources'][n]={'requested':s['requested'],'config':H.record(s['config_file']),'weights':H.record(s['model_file'])}
 M['shape_normalization']=p.shape_slat_normalization;M['shape_sampler']=p.shape_slat_sampler_params;M['image_extractor']=type(p.image_cond_model).__name__;M['low_vram']=p.low_vram
 ckpt=ROOT/'results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt';M['fixed_checkpoint']=H.record(ckpt)
 assets=[('0086bbab37559442',0),('0fab4af3f9a73bdd',6)]
 sampler=p.shape_slat_sampler.sample;current={}
 def noise_capture(model,noise,*args,**kw):
  current['noise_sha256']=tensorhash(noise.feats);current['noise_coords_sha256']=ah(noise.coords.detach().cpu().numpy());return sampler(model,noise,*args,**kw)
 p.shape_slat_sampler.sample=noise_capture
 for tag in ['official','fixed_shape20k']:
  if tag=='fixed_shape20k':
   model=p.models['shape_slat_flow_model_1024'];replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.)
   state=torch.load(ckpt,map_location='cpu',weights_only=True);state.pop('rope_phases',None);validate_lora(model,replaced,state,8,8.,0.);checked_load(model,state,allow_rope=True);del state
   M['fixed_lora']={'adapters':len(replaced),'rank':8,'alpha':8,'dropout':0.,'eval':not model.training,'adapter_dtype':str(next(x for n,x in model.named_parameters() if 'lora_down' in n).dtype)}
  for obj,view in assets:
   print('CASE_START',tag,obj,flush=True);d=OUT/f'{obj}_{tag}';d.mkdir();r={'object':obj,'tag':tag,'status':'running','view_index':view,'outputs':{}};M['results'].append(r);save()
   sp=ROOT/'diagnostics/remediation_20261013/common_shape_pilot'/obj/'ss_support64_from_shape_encoder.npz'
   with np.load(sp) as a:c=a['coords'].astype(np.int32)
   coords=torch.from_numpy(np.column_stack([np.zeros(len(c),dtype=np.int32),c])).to('cuda')
   r['support']=H.record(sp);r['coords_sha256']=ah(coords.cpu().numpy());r['token_count']=len(c)
   images=sorted((ROOT/'training_data/multiview_1186/renders_cond_t'/obj).glob(f'{view:03d}_*.png'));assert len(images)==1
   r['image']=H.record(images[0]);r['transform_metadata']=H.record(images[0].parent/'transforms.json')
   with Image.open(images[0]) as im:
    im=im.convert('RGBA');alpha=np.asarray(im)[:,:,3];assert np.any(alpha<255) and np.any(alpha>204)
    pre=p.preprocess_image(im)
   pre.save(d/'condition_black.png');r['outputs']['condition']=H.record(d/'condition_black.png')
   seed();t=time.monotonic();cond512=p.get_cond([pre],512);cond1024=p.get_cond([pre],1024);sscond=p.get_ss_cond([pre]);r['condition_seconds']=stamp('COND',t)
   r['cond512_sha256']=tensorhash(cond512['cond']);r['cond1024_sha256']=tensorhash(cond1024['cond']);r['sscond_sha256']=tensorhash(sscond['cond']);del cond512,sscond
   r['cpu_rng_before_noise_sha256']=ah(torch.get_rng_state().numpy());current.clear();t=time.monotonic()
   with torch.no_grad():latent=p.sample_shape_slat(cond1024,p.models['shape_slat_flow_model_1024'],coords)
   r['flow_seconds']=stamp('FLOW',t);r.update(current);r['latent_finite']=bool(torch.isfinite(latent.feats).all());assert r['latent_finite'];r['latent_shape']=list(latent.feats.shape)
   np.savez_compressed(d/'sampled_shape_mean.npz',coords=latent.coords.cpu().numpy(),feats=latent.feats.detach().cpu().numpy());r['outputs']['latent']=H.record(d/'sampled_shape_mean.npz');del cond1024
   t=time.monotonic()
   with torch.no_grad():meshes,subs=p.decode_shape_slat(latent,1024)
   r['decode_seconds']=stamp('DECODE',t);mesh=meshes[0];v=mesh.vertices.detach().cpu().float().numpy();f=mesh.faces.detach().cpu().numpy();assert np.isfinite(v).all() and len(f)>0
   np.savez_compressed(d/'raw_geometry.npz',vertices=v,faces=f);r['outputs']['raw_npz']=H.record(d/'raw_geometry.npz')
   import trimesh
   trimesh.Trimesh(v,f,process=False).export(d/'raw_geometry.ply');vf=np.column_stack((v[:,0],v[:,2],-v[:,1]));trimesh.Trimesh(vf,f,process=False).export(d/'formal_geometry_only.ply');r['outputs']['raw_ply']=H.record(d/'raw_geometry.ply');r['outputs']['formal_ply']=H.record(d/'formal_geometry_only.ply');r['vertices']=len(v);r['faces']=len(f);r['status']='success';save()
   del latent,meshes,subs,mesh,coords;torch.cuda.empty_cache();print('CASE_DONE',tag,obj,len(v),len(f),flush=True)
 for obj,_ in assets:
  pair=[x for x in M['results'] if x['object']==obj];assert len(pair)==2
  fields=['coords_sha256','token_count','cond512_sha256','cond1024_sha256','sscond_sha256','cpu_rng_before_noise_sha256','noise_sha256','noise_coords_sha256']
  checks={k:pair[0][k]==pair[1][k] for k in fields};assert all(checks.values()),checks
  M.setdefault('single_variable_checks',{})[obj]=checks
 M.update(status='success',finished_unix=time.time());save();print('COMPAT_FLOW_COMPLETE',flush=True)
except Exception as e:
 M.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),finished_unix=time.time());save();traceback.print_exc();sys.exit(1)
