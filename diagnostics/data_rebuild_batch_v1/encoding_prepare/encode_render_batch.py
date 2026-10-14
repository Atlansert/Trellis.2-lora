"""Prepared only: validate first, then staged six-asset shape/SS re-encoding."""
import argparse,json,sys,time,gc
from pathlib import Path
import numpy as np
from batch_encoding_contract import validate_batch,fresh_or_resume,sha,atomic,load_geometry,stage_done,stage_commit,assert_new,device_guard
ROOT=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--batch-validation','--validation',dest='batch_validation',required=True);p.add_argument('--output');p.add_argument('--stage',choices=['validate','encode','decode','ss','finalize'],default='validate');a=p.parse_args()
review,rows=validate_batch(a.batch_validation)
if a.stage=='validate':print(json.dumps({'assets':len(rows),'input_validation_passed':True,'models_loaded':False,'training_approved':False},indent=2));sys.exit(0)
if not a.output:raise ValueError('output required')
device_guard(a.stage);out=fresh_or_resume(a.output,a.batch_validation)
for row in rows:(out/row['object_id']).mkdir(exist_ok=True)
def weights(prefix):
 return {'prefix':str(prefix),'architecture':json.loads(Path(str(prefix)+'.json').read_text()),'config_sha256':sha(str(prefix)+'.json'),'weight_sha256':sha(str(prefix)+'.safetensors')}
def moments(v,f):
 tri=v[f].astype(np.float64);ar=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;cent=tri.mean(1);mu=np.average(cent,axis=0,weights=ar);q=cent-mu;cov=(q*ar[:,None]).T@q/ar.sum();within=tri-cent[:,None,:];cov+=np.einsum('n,nki,nkj->ij',ar,within,within)/(12*ar.sum());ev,vec=np.linalg.eigh(cov);axis=vec[:,-1]
 return {'extents':np.ptp(v,axis=0).tolist(),'lengthY_offangle_degrees':float(np.degrees(np.arccos(np.clip(abs(axis[1]),0,1)))),'surface_integrated_covariance':cov.tolist(),'eigenvalues':ev.tolist(),'method':'exact triangle surface integral incl within-triangle moments'}
if a.stage in ('encode','decode'):
 import torch
 from trellis2 import models
 from trellis2.modules import sparse as sp
 torch.set_num_threads(8);torch.manual_seed(42)
 prefix=ROOT/'checkpoints/ckpts'/('shape_enc_next_dc_f16c32_fp16' if a.stage=='encode' else 'shape_dec_next_dc_f16c32_fp16')
 checkpoint=weights(prefix);model=models.from_pretrained(str(prefix)).eval().cuda()
 if a.stage=='decode':model.set_resolution(1024)
 with torch.inference_mode():
  for row in rows:
   folder=out/row['object_id'];geometry=Path(row['_geometry_path']);gh=sha(geometry)
   if stage_done(folder,a.stage,gh):print('VERIFIED_SKIP',a.stage,row['object_id'],flush=True);continue
   started=time.perf_counter();v,f=load_geometry(geometry)
   if a.stage=='encode':
    import o_voxel
    paths=[folder/'dual_grid1024.npz',folder/'shape_mean.npz',folder/'ss_support64.npz'];assert_new(paths)
    c,d,i=o_voxel.convert.mesh_to_flexible_dual_grid(torch.from_numpy(v).float(),torch.from_numpy(f).long(),grid_size=1024,aabb=[[-.5]*3,[.5]*3],face_weight=1.,boundary_weight=.2,regularization_weight=1e-2,timing=False)
    local=d*1024-c
    if not torch.isfinite(local).all() or local.min()<-.001 or local.max()>1.001:raise ValueError('dual vertices not finite/local')
    quant=(local.clamp(0,1)*255).to(torch.uint8);inp=sp.SparseTensor(quant.float()/255,torch.cat([torch.zeros_like(c[:,:1]),c],-1).int()).cuda();inter=inp.replace(i.bool().cuda())
    z,mean,logvar=model(inp,inter,sample_posterior=False,return_raw=True);torch.cuda.synchronize();coords=z.coords[:,1:].cpu().numpy().astype(np.int32);feats=z.feats.float().cpu().numpy()
    if feats.shape!=(len(coords),32) or not np.isfinite(feats).all() or coords.min()<0 or coords.max()>63 or not torch.equal(z.feats,mean):raise ValueError('shape posterior invalid')
    if len(np.unique(coords,axis=0))!=len(coords):raise ValueError('duplicate shape coords')
    down=np.unique(c.numpy()//16,axis=0)
    if {tuple(x) for x in down}!={tuple(x) for x in coords}:raise ValueError('shape support differs from dualgrid coarsening')
    np.savez_compressed(paths[0],coords=c.numpy(),vertices=quant.numpy(),intersected=i.numpy());np.savez_compressed(paths[1],coords=coords.astype(np.uint8),feats=feats,mean=feats);np.savez_compressed(paths[2],coords=coords)
    result={'checkpoint':checkpoint,'tokens':len(coords),'dualgrid_cells':len(c),'mean_shape':list(feats.shape),'dtype':'float32','coords_disk_dtype':'uint8','posterior_sample':False,'shared_support_constructed_not_independent':True,'source_geometry_sha256':gh}
    del inp,inter,z,mean,logvar,c,d,i,local,quant
   else:
    if not stage_done(folder,'encode',gh):raise ValueError('shape encoding stage not committed')
    paths=[folder/'decoded_shape_geometry.npz',folder/'decoded_shape_geometry.ply'];assert_new(paths)
    with np.load(folder/'shape_mean.npz',allow_pickle=False) as z:coords=z['coords'].astype(np.int32);feats=z['feats'].copy()
    x=sp.SparseTensor(torch.from_numpy(feats),torch.from_numpy(np.column_stack([np.zeros(len(coords),np.int32),coords]))).cuda();mesh=model(x)[0];torch.cuda.synchronize();dv=mesh.vertices.float().cpu().numpy();df=mesh.faces.cpu().numpy().astype(np.int32)
    if not np.isfinite(dv).all() or not len(df):raise ValueError('decoded geometry invalid')
    import trimesh,importlib.util
    decoded=trimesh.Trimesh(dv,df,process=False);source=trimesh.Trimesh(v,f,process=False);np.savez_compressed(paths[0],vertices=dv,faces=df);decoded.export(paths[1]);surface={}
    if importlib.util.find_spec('rtree') is None:raise RuntimeError('exact triangle proximity requires existing rtree; no dependency install/fallback claim')
    for label,src,dst in [('source_to_decode',source,decoded),('decode_to_source',decoded,source)]:
     pts,_=trimesh.sample.sample_surface(src,512,seed=42);_,dist,_=trimesh.proximity.closest_point(dst,pts);surface[label]={'mean':float(dist.mean()),'p95':float(np.quantile(dist,.95)),'max':float(dist.max())}
    result={'checkpoint':checkpoint,'vertices':len(dv),'faces':len(df),'finite':True,'source_moments':moments(v,f),'decoded_moments':moments(dv,df),'surface_error':surface,'symmetric_mean_surface_distance':sum(d['mean'] for d in surface.values())/2,'surface_method':'512 area samples eachside to exact triangle distance, no alignment/scale/rotation fitting','roof_nose_wheel_semantics':'not proven automatically through VAE','formal_glb_acceptance':False}
    del x,mesh
   result['elapsed_seconds']=time.perf_counter()-started;stage_commit(folder,a.stage,gh,paths,result);gc.collect();torch.cuda.empty_cache();print('STAGE_ASSET_COMPLETE',a.stage,row['object_id'],result['elapsed_seconds'],flush=True)
elif a.stage=='ss':
 import torch
 from trellis2 import models
 torch.set_num_threads(8)
 ep=Path('/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16');dp=Path('/home/test/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16')
 enc=models.from_pretrained(str(ep));dec=models.from_pretrained(str(dp))
 for model in [enc,dec]:model.convert_to_fp32();model.float().cpu().eval()
 with torch.inference_mode():
  for row in rows:
   folder=out/row['object_id'];gh=sha(row['_geometry_path'])
   if stage_done(folder,'ss',gh):print('VERIFIED_SKIP ss',row['object_id'],flush=True);continue
   if not stage_done(folder,'encode',gh):raise ValueError('shape support not committed')
   paths=[folder/'ss_mean.npz',folder/'decoded_ss_support64.npz'];assert_new(paths)
   with np.load(folder/'ss_support64.npz',allow_pickle=False) as z:coords=z['coords'].astype(int)
   vol=torch.zeros((1,1,64,64,64));vol[0,0,coords[:,0],coords[:,1],coords[:,2]]=1.;mean=enc(vol,sample_posterior=False);logits=dec(mean)
   if mean.shape!=(1,8,16,16,16) or not torch.isfinite(mean).all() or not torch.isfinite(logits).all():raise ValueError('SS mean/decoder invalid')
   pred=logits>0;gt=vol>0;iou=(pred & gt).sum().item()/(pred|gt).sum().item();dc=torch.argwhere(pred[0,0]).numpy().astype(np.int32)
   np.savez_compressed(paths[0],mean=mean[0].numpy());np.savez_compressed(paths[1],coords=dc)
   stage_commit(folder,'ss',gh,paths,{'encoder':weights(ep),'decoder':weights(dp),'tokens':len(coords),'decode_tokens':len(dc),'support_iou':iou,'mean_shape':[8,16,16,16],'mean_dtype':'float32','finite':True,'target_axes':'direct raw(depth,length,up), NO legacy permutation'})
   print('STAGE_ASSET_COMPLETE ss',row['object_id'],iou,flush=True)
else:
 if (out/'COMPLETE.json').exists():
  prior=json.loads((out/'COMPLETE.json').read_text())
  if sha(out/'manifest.json')!=prior['manifest_sha256']:raise ValueError('final manifest changed')
  raise FileExistsError('batch already finalized; preserved without overwrite')
 records=[]
 for row in rows:
  folder=out/row['object_id'];gh=sha(row['_geometry_path']);stages={}
  for stage in ['encode','decode','ss']:
   if not stage_done(folder,stage,gh):raise ValueError('missing stage commit')
   stages[stage]=json.loads((folder/(stage+'_COMPLETE.json')).read_text())
  records.append({'object_id':row['object_id'],'geometry_sha256':gh,'render_manifest_sha256':row['render_manifest_sha256'],'initial_training_condition_view':0,'other_views':'audit only, not blindly shared canonical targets','stages':stages})
 atomic(out/'manifest.json',{'schema':'trellis.render-batch-encoded.v1','assets':records,'batch_validation_sha256':sha(a.batch_validation),'training_approved':False,'pbr_encoded':False,'weights_not_updated':True,'encode_script_sha256':sha(__file__)})
 atomic(out/'COMPLETE.json',{'manifest_sha256':sha(out/'manifest.json'),'assets':6,'training_approved':False});print('ALL_SIX_ENCODING_STAGES_COMPLETE_NOT_TRAINING_APPROVAL',flush=True)
