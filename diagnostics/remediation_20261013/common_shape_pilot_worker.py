import os,sys,json,time,hashlib,gc
from pathlib import Path
import numpy as np
ROOT=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=ROOT/'diagnostics/remediation_20261013/common_shape_pilot'
SRC=ROOT/'diagnostics/remediation_20261013/common_frame_pilot'
IDS=['0086bbab37559442','0fab4af3f9a73bdd']

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def freshjson(p,v):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(v,indent=2))
def stamp(s):print(s,time.time(),flush=True)
def inputmesh(obj):
 man=json.loads((SRC/'manifest.json').read_text());r=next(x for x in man['assets'] if x['object']==obj)
 p=SRC/obj/'proper_geometry.npz';assert digest(p)==r['geometry_sha256'];assert r['determinant']==1.
 with np.load(p,allow_pickle=False) as z:v=z['vertices'].copy();f=z['faces'].copy()
 assert np.isfinite(v).all() and v.min()>=-.5 and v.max()<=.5
 return v,f,r

def encode():
 import torch,o_voxel
 from trellis2 import models
 from trellis2.modules import sparse as sp
 torch.set_num_threads(8);torch.manual_seed(42)
 base=ROOT/'checkpoints/ckpts/shape_enc_next_dc_f16c32_fp16'
 stamp('LOAD_ENCODER_START');enc=models.from_pretrained(str(base)).eval().cuda();stamp('LOAD_ENCODER_END')
 rec={'architecture':json.loads(base.with_suffix('.json').read_text()),'config_sha256':digest(base.with_suffix('.json')),'weights_sha256':digest(base.with_suffix('.safetensors')),'checkpoint_prefix':str(base),'sample_posterior':False,'resolution':1024,'assets':[]}
 with torch.inference_mode():
  for obj in IDS:
   stamp(obj+' DUALGRID_START');t=time.perf_counter();folder=OUT/obj;folder.mkdir(exist_ok=False)
   v,f,r=inputmesh(obj)
   c,d,i=o_voxel.convert.mesh_to_flexible_dual_grid(torch.from_numpy(v).float(),torch.from_numpy(f).long(),grid_size=1024,aabb=[[-.5]*3,[.5]*3],face_weight=1.,boundary_weight=.2,regularization_weight=1e-2,timing=False)
   local=d*1024-c;assert torch.isfinite(local).all() and local.min()>=-.001 and local.max()<=1.001
   quantized=(local.clamp(0,1)*255).to(torch.uint8);inp=sp.SparseTensor(quantized.float()/255,torch.cat([torch.zeros_like(c[:,:1]),c],-1).int()).cuda();inter=inp.replace(i.bool().cuda())
   np.savez_compressed(folder/'new_dual_grid1024.npz',coords=c.numpy(),vertices=quantized.numpy(),intersected=i.numpy())
   stamp(obj+' ENCODE_START');z,mean,logvar=enc(inp,inter,sample_posterior=False,return_raw=True);torch.cuda.synchronize()
   coords=z.coords[:,1:].cpu().numpy().astype(np.int32);feats=z.feats.float().cpu().numpy();assert feats.shape==(len(coords),32) and np.isfinite(feats).all() and coords.min()>=0 and coords.max()<=63
   np.savez_compressed(folder/'new_shape_mean.npz',coords=coords.astype(np.uint8),feats=feats,mean=feats)
   support=np.unique(coords,axis=0);assert len(support)==len(coords);np.savez_compressed(folder/'ss_support64_from_shape_encoder.npz',coords=support)
   with np.load(SRC/obj/'ss_support64.npz') as z0:old=z0['coords']
   a={tuple(x) for x in support.tolist()};b={tuple(x) for x in old.tolist()};iou=len(a&b)/len(a|b)
   down=np.unique(c.numpy()//16,axis=0);ds={tuple(x) for x in down.tolist()}
   ar={'object':obj,'source_geometry_sha256':r['geometry_sha256'],'dual_grid_voxels':len(c),'dualgrid_vs_encoder_support_iou':len(ds&a)/len(ds|a),'token_count':len(coords),'coords_grid':64,'coords_dtype':str(coords.dtype),'mean_shape':list(feats.shape),'mean_dtype':str(feats.dtype),'mean_finite':True,'mean_matches_return_raw_posterior':bool(torch.equal(z.feats,mean)),'shape_encoder_vs_independent_open3d64_iou':iou,'encoder_grid64_spans':(np.ptp(coords,axis=0)+1).tolist(),'ss_derived_shared_support_equality':True,'shared_equality_is_constructed_not_independent':True,'elapsed_seconds':time.perf_counter()-t,'input_up_marker':r['raw_up_marker'],'decoded_semantic_roof_wheel_unverified':True,'hashes':{x.name:digest(x) for x in folder.iterdir() if x.is_file()}}
   rec['assets'].append(ar);stamp(obj+' ENCODE_END');del inp,inter,z,mean,logvar,c,d,i,local,quantized;gc.collect();torch.cuda.empty_cache()
 freshjson(OUT/'encode_result.json',rec)

def pca(v,f,axis):
 tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;cent=tri.mean(1);mu=np.average(cent,axis=0,weights=area);q=cent-mu;cov=(q*area[:,None]).T@q/area.sum();w,u=np.linalg.eigh(cov);a=u[:,-1];angle=float(np.degrees(np.arccos(np.clip(abs(a[axis]),0,1))))
 return {'extents':np.ptp(v,axis=0).tolist(),'major_axis':a.tolist(),'offangle_degrees':angle,'surface_area':float(area.sum()),'eigenvalues':w.tolist()}

def decode():
 import torch,trimesh,importlib.util
 from trellis2 import models
 from trellis2.modules import sparse as sp
 from data_toolkit.canonical_frame import internal_to_formal_glb
 torch.set_num_threads(8);torch.manual_seed(42)
 base=ROOT/'checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16';stamp('LOAD_DECODER_START');dec=models.from_pretrained(str(base)).eval().cuda();dec.set_resolution(1024);stamp('LOAD_DECODER_END')
 rec={'architecture':json.loads(base.with_suffix('.json').read_text()),'config_sha256':digest(base.with_suffix('.json')),'weights_sha256':digest(base.with_suffix('.safetensors')),'checkpoint_prefix':str(base),'resolution':1024,'assets':[]}
 with torch.inference_mode():
  for obj in IDS:
   t=time.perf_counter();folder=OUT/obj;stamp(obj+' DECODE_START')
   with np.load(folder/'new_shape_mean.npz') as z:c=z['coords'].astype(np.int32);feat=z['feats'].copy()
   x=sp.SparseTensor(torch.from_numpy(feat),torch.from_numpy(np.column_stack([np.zeros(len(c),np.int32),c]))).cuda();mesh=dec(x)[0];torch.cuda.synchronize();v=mesh.vertices.float().cpu().numpy();f=mesh.faces.cpu().numpy().astype(np.int32)
   assert len(v)>0 and len(f)>0 and np.isfinite(v).all();target=folder/'decoded_raw_geometry.ply';assert not target.exists();decoded=trimesh.Trimesh(v,f,process=False);decoded.export(target)
   np.savez_compressed(folder/'decoded_raw_geometry.npz',vertices=v,faces=f)
   stamp(obj+' QUALITY_START');sv,sf,r=inputmesh(obj);source=trimesh.Trimesh(sv,sf,process=False)
   quality={};samples=512
   if importlib.util.find_spec('rtree') is not None:
    for label,aa,bb in [('source_to_decode',source,decoded),('decode_to_source',decoded,source)]:
     pts,_=trimesh.sample.sample_surface(aa,samples,seed=42);_,dist,_=trimesh.proximity.closest_point(bb,pts)
     quality[label]={'samples':samples,'mean':float(dist.mean()),'rmse':float(np.sqrt(np.mean(dist**2))),'p50_p95_max':np.quantile(dist,[.5,.95,1]).tolist()}
    quality['method']='Area-uniform same-frame surface samples512/side to exact closest triangle surface; no fitting/alignment; normalized longlength approximately1'
   else:
    from scipy.spatial import cKDTree
    sa,_=trimesh.sample.sample_surface(source,100000,seed=42);da,_=trimesh.sample.sample_surface(decoded,100000,seed=42)
    for label,aa,bb in [('source_to_decode',sa,da),('decode_to_source',da,sa)]:
     dist=cKDTree(bb).query(aa[:samples])[0];quality[label]={'samples':samples,'mean':float(dist.mean()),'rmse':float(np.sqrt(np.mean(dist**2))),'p50_p95_max':np.quantile(dist,[.5,.95,1]).tolist()}
    quality['method']='Sampled-surface KD nearest proxy100000 reference samples, not exact triangle distance; rtree unavailable, no dependencies installed'
   quality['symmetric_mean_distance']=sum(quality[k]['mean'] for k in ['source_to_decode','decode_to_source'])/2
   coarse=np.clip(np.floor((v+.5)*64),0,63).astype(np.int32);ca={tuple(x) for x in coarse.tolist()};enc={tuple(x) for x in c.tolist()}
   rec['assets'].append({'object':obj,'vertices':len(v),'faces':len(f),'finite':True,'raw_metrics':pca(v,f,1),'formal_rotation_diagnostic_metrics':pca(internal_to_formal_glb(v),f,2),'source_raw_metrics':pca(sv,sf,1),'same_asset_surface_error':quality,'decoded_vertex_coarse_support_tokens':len(ca),'decoded_vertex_coarse_vs_encoder_support_iou':len(ca&enc)/len(ca|enc),'decoded_vertices_within_encoder_cells_fraction':sum(tuple(x) in enc for x in coarse.tolist())/len(coarse),'note_support':'Decoded mesh vertices may lie in neighboring parent cell due local halfvoxel margin; this coarse vertex support is not voxelized surface occupancy','up_semantic_roof_wheel':'UNVERIFIED: coordinate marker is not propagated semantic label through VAE','formal_output':'Rotation-only diagnostic, not GLB/remesh/textured acceptance','elapsed_seconds':time.perf_counter()-t,'hashes':{x.name:digest(x) for x in folder.iterdir() if x.is_file()}});stamp(obj+' DECODE_END');del x,mesh;gc.collect();torch.cuda.empty_cache()
 freshjson(OUT/'decode_result.json',rec)

if __name__=='__main__':
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5';assert OUT.is_dir();mode=sys.argv[1]
  if mode=='encode':encode()
  elif mode=='decode':decode()
  else:raise ValueError(mode)
 except Exception as e:
  import traceback;traceback.print_exc();freshjson(OUT/(sys.argv[1]+'_error.json'),{'error_type':type(e).__name__,'error':str(e),'training_approved':False});sys.exit(1)
 def noop():pass
 stamp('WORKER_ENDED_SUCCESS')
 torch=None
 gc.collect()
