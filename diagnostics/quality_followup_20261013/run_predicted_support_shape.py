import os,sys,json,time,hashlib,copy,traceback
from pathlib import Path
R=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(R));os.chdir(R)
import numpy as np,torch
from PIL import Image
import trimesh
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers
from scripts.utils.inference_contract import HashCache,TARGETS,checked_load,validate_lora,write_manifest
B=R/'diagnostics/quality_followup_20261013';O=B/'predicted_support_shape';O.mkdir(exist_ok=False)
H=HashCache();M={'schema':'trellis.predicted-support-shape.v1','status':'running','seed':42,'no_training':True,'no_texture':True,'formal_acceptance_eligible':False,'cases':[],'ss_samples':[]}
def th(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def save():write_manifest(O/'manifest.json',M)
def rng():return (torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()])
def restore(s):torch.set_rng_state(s[0]);torch.cuda.set_rng_state_all(s[1])
def install(model,path):
 names=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.)
 state=torch.load(path,map_location='cpu',weights_only=True);state.pop('rope_phases',None);validate_lora(model,names,state,8,8.,0.);checked_load(model,state);del state
 return names
try:
 assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
 M['source']=H.record(__file__);p=Trellis2ImageTo3DPipeline.from_pretrained(str(R/'checkpoints'));p.cuda();assert not p.ss_swap_yz and not p.ss_canonicalize
 M['base_sources']={k:{'config':H.record(v._checkpoint_source['config_file']),'weights':H.record(v._checkpoint_source['model_file'])} for k,v in p.models.items()}
 shape_base=p.models['shape_slat_flow_model_1024'];shape_fixed=copy.deepcopy(shape_base)
 fixedpath=R/'results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt';install(shape_fixed,fixedpath);M['shape_fixed']=H.record(fixedpath)
 sspath=B/'short_ss_proper_50/ckpts/denoiser_step0000050.pt';M['ss_proper_live50']=H.record(sspath)
 M['samplers']={'ss':p.sparse_structure_sampler_params,'shape':p.shape_slat_sampler_params};M['shape_normalization']=p.shape_slat_normalization
 plan=json.loads((B/'short_ss_pilot_plan/manifest.json').read_text());ob=next(x for x in plan['objects'] if x['object']=='0086bbab37559442')
 inputs=[('train0086',Path(ob['condition']['path']),False),('004',R/'data_test/004.png',True)]
 capture={};sampler=p.shape_slat_sampler.sample
 def capturing(model,noise,*args,**kwargs):
  capture['noise_sha256']=th(noise.feats);capture['noise_coords_sha256']=th(noise.coords);return sampler(model,noise,*args,**kwargs)
 p.shape_slat_sampler.sample=capturing
 for ss_tag in ['official','proper_live50']:
  if ss_tag=='proper_live50':install(p.models['sparse_structure_flow_model'],sspath)
  for name,imagepath,preprocess in inputs:
   with Image.open(imagepath) as im:im=p.preprocess_image(im.convert('RGBA')) if preprocess else im.convert('RGB')
   torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   with torch.inference_mode():
    c512=p.get_cond([im],512);c1024=p.get_cond([im],1024);css=p.get_ss_cond([im]);before_ss=rng();coords=p.sample_sparse_structure(css,64,1,{});after_ss=rng()
   assert coords.shape[1]==4 and len(coords)>0
   np.savez_compressed(O/f'ss_{ss_tag}_{name}.npz',coords=coords.cpu().numpy());im.save(O/f'condition_{ss_tag}_{name}.png')
   ssrec={'ss':ss_tag,'input':name,'input_file':H.record(imagepath),'condition':H.record(O/f'condition_{ss_tag}_{name}.png'),'cond512_sha256':th(c512['cond']),'cond1024_sha256':th(c1024['cond']),'sscond_sha256':th(css['cond']),'cpu_rng_before_ss':th(before_ss[0]),'cuda_rng_before_ss':[th(x) for x in before_ss[1]],'coords':H.record(O/f'ss_{ss_tag}_{name}.npz'),'coords_sha256':th(coords),'count':len(coords),'spans':(np.ptp(coords.cpu().numpy()[:,1:],axis=0)+1).tolist()}
   if name=='train0086':
    gt=np.load(R/'diagnostics/remediation_20261013/common_shape_pilot/0086bbab37559442/ss_support64_from_shape_encoder.npz')['coords'];encode=lambda c:set(map(tuple,c.tolist()));x=encode(coords.cpu().numpy()[:,1:]);y=encode(gt);ssrec['proper_gt_support_iou']=len(x&y)/len(x|y)
   else:ssrec['no_gt']=True
   M['ss_samples'].append(ssrec);save()
   for shape_tag,flow in [('official',shape_base),('fixed20k',shape_fixed)]:
    restore(after_ss);capture.clear();d=O/f'{name}_ss_{ss_tag}_shape_{shape_tag}';d.mkdir();rec={'input':name,'ss':ss_tag,'shape':shape_tag,'coords_sha256':th(coords),'cpu_rng_before_shape':th(torch.get_rng_state()),'cuda_rng_before_shape':[th(x) for x in torch.cuda.get_rng_state_all()],'outputs':{}}
    t=time.monotonic()
    with torch.inference_mode():latent=p.sample_shape_slat(c1024,flow,coords);meshes,subs=p.decode_shape_slat(latent,1024)
    torch.cuda.synchronize();rec['seconds']=time.monotonic()-t;rec.update(capture)
    v=meshes[0].vertices.detach().cpu().float().numpy();f=meshes[0].faces.detach().cpu().numpy();assert np.isfinite(v).all() and len(f)>0
    np.savez_compressed(d/'raw_geometry.npz',vertices=v,faces=f);np.savez_compressed(d/'shape_latent.npz',coords=latent.coords.cpu().numpy(),feats=latent.feats.detach().cpu().float().numpy())
    trimesh.Trimesh(v,f,process=False).export(d/'raw_geometry.ply');formal=v[:,[0,2,1]]*np.array([1,1,-1]);trimesh.Trimesh(formal,f,process=False).export(d/'formal_geometry_only.ply')
    for fname in ['raw_geometry.npz','shape_latent.npz','raw_geometry.ply','formal_geometry_only.ply']:rec['outputs'][fname]=H.record(d/fname)
    rec['vertices']=len(v);rec['faces']=len(f);rec['status']='success';M['cases'].append(rec);save();print('CASE_DONE',name,ss_tag,shape_tag,rec['seconds'],flush=True)
    del latent,meshes,subs;torch.cuda.empty_cache()
   del coords,c512,c1024,css
 for ss in ['official','proper_live50']:
  for name,_,_ in inputs:
   a,b=[x for x in M['cases'] if x['input']==name and x['ss']==ss]
   for k in ['coords_sha256','cpu_rng_before_shape','cuda_rng_before_shape','noise_sha256','noise_coords_sha256']:assert a[k]==b[k],k
 for name,_,_ in inputs:
  a,b=[x for x in M['ss_samples'] if x['input']==name]
  for k in ['sscond_sha256','cond512_sha256','cond1024_sha256','cpu_rng_before_ss','cuda_rng_before_ss']:assert a[k]==b[k],k
 M.update(status='success',same_noise_within_each_shape_pair=True,same_conditions_and_rng_between_ss_models=True);save();(O/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':H.record(O/'manifest.json')['sha256'],'training_approved':False},indent=2));print('EIGHT_CASES_COMPLETE',flush=True)
except Exception as e:
 M.update(status='failed',error=str(e),traceback=traceback.format_exc());save();raise
