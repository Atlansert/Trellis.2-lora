import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,time,gc
from pathlib import Path
root=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(root));os.chdir(root)
import numpy as np,torch
from trellis2 import models
b=root/'diagnostics/data_rebuild_batch_v1';enc=b/'encoded_targets_v1';out=b/'target_validation';out.mkdir(exist_ok=False);started=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4194304),b''):h.update(c)
 return h.hexdigest()
def moment(v,f):
 total=0.;s=np.zeros(3);q=np.zeros((3,3))
 for k in range(0,len(f),65536):
  tri=v[f[k:k+65536]].astype(float);ar=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;c=tri.mean(1);w=tri-c[:,None,:];total+=ar.sum();s+=(c*ar[:,None]).sum(0);q+=(c*ar[:,None]).T@c+np.einsum('n,nki,nkj->ij',ar,w,w)/12
 mu=s/total;cov=q/total-np.outer(mu,mu);ev,vec=np.linalg.eigh(cov);return cov,float(np.degrees(np.arccos(np.clip(abs(vec[1,-1]),0,1))))
m=json.loads((enc/'manifest.json').read_text());complete=json.loads((enc/'COMPLETE.json').read_text());assert sha(enc/'manifest.json')==complete['manifest_sha256'];validation=json.loads((b/'batch_render_validation.json').read_text());assert sha(b/'batch_render_validation.json')==m['batch_validation_sha256'];renders={r['object_id']:Path(r['render_directory']) for r in validation['objects']}
weights={};rows=[]
for asset in m['assets']:
 oid=asset['object_id'];folder=enc/oid;render=renders[oid];rm=json.loads((render/'manifest.json').read_text());assert sha(render/'manifest.json')==asset['render_manifest_sha256'];assert sha(render/'canonical_geometry.npz')==asset['geometry_sha256'];assert sha(rm['source_path'])==rm['source_sha256'];assert sha(rm['review_path'])==rm['review_sha256'];assert asset['initial_training_condition_view']==0
 for stage,record in asset['stages'].items():
  assert record['input_geometry_sha256']==asset['geometry_sha256']
  assert json.loads((folder/(stage+'_COMPLETE.json')).read_text())==record
  for name,h in record['files'].items():assert sha(folder/name)==h,(oid,name)
  for key in ['checkpoint','encoder','decoder']:
   cp=record['result'].get(key)
   if cp:weights[cp['prefix']]=cp
 with np.load(folder/'shape_mean.npz',allow_pickle=False) as z:
  c=z['coords'];feat=z['feats'];mean=z['mean'];assert c.dtype==np.uint8 and c.ndim==2 and c.shape[1]==3 and c.max()<64;assert len(np.unique(c,axis=0))==len(c);assert feat.dtype==np.float32 and feat.shape==(len(c),32) and np.isfinite(feat).all() and np.array_equal(feat,mean)
 with np.load(folder/'ss_support64.npz',allow_pickle=False) as z:shared=z['coords'];assert np.array_equal(c.astype(int),shared.astype(int))
 with np.load(folder/'dual_grid1024.npz',allow_pickle=False) as z:
  dc=z['coords'];dv=z['vertices'];ix=z['intersected'];assert dc.ndim==2 and dc.shape[1]==3 and dc.min()>=0 and dc.max()<1024;assert dv.dtype==np.uint8 and dv.shape==(len(dc),3);assert np.isfinite(dc).all();assert np.array_equal(np.unique(dc//16,axis=0),np.unique(c.astype(int),axis=0))
 with np.load(folder/'ss_mean.npz',allow_pickle=False) as z:ss=z['mean'];assert ss.shape==(8,16,16,16) and ss.dtype==np.float32 and np.isfinite(ss).all()
 with np.load(folder/'decoded_ss_support64.npz',allow_pickle=False) as z:sd=z['coords'];assert sd.min()>=0 and sd.max()<64
 gtset={tuple(p) for p in shared};savedset={tuple(p) for p in sd};saved_iou=len(gtset&savedset)/len(gtset|savedset)
 with np.load(render/'canonical_geometry.npz',allow_pickle=False) as z:v=z['vertices'];f=z['faces'];cov,angle=moment(v,f);ext=np.ptp(v,axis=0)
 with np.load(folder/'decoded_shape_geometry.npz',allow_pickle=False) as z:v2=z['vertices'];f2=z['faces'];assert np.isfinite(v2).all() and np.issubdtype(f2.dtype,np.integer) and f2.min()>=0 and f2.max()<len(v2);cov2,angle2=moment(v2,f2);ext2=np.ptp(v2,axis=0)
 result=asset['stages']['decode']['result'];assert np.allclose(cov,result['source_moments']['surface_integrated_covariance'],atol=1e-10);assert np.allclose(cov2,result['decoded_moments']['surface_integrated_covariance'],atol=1e-10)
 rows.append({'object_id':oid,'shape_tokens':len(c),'ss_saved_roundtrip_iou':saved_iou,'extents_source':ext.tolist(),'extents_decoded':ext2.tolist(),'relative_extent_error':((ext2-ext)/ext).tolist(),'source_length_angle':angle,'decoded_length_angle':angle2,'symmetric_sampled_exact_triangle_distance':result['symmetric_mean_surface_distance'],'reported_max_sampled_surface_distance':max(x['max'] for x in result['surface_error'].values()),'mean':ss.copy(),'shared':shared.copy(),'saved_decoded':sd.copy()});print('VERIFIED_ARRAYS_HASHES_MOMENTS',oid,flush=True);del v,f,v2,f2,cov,cov2;gc.collect()
for prefix,cp in weights.items():assert sha(prefix+'.json')==cp['config_sha256'];assert sha(prefix+'.safetensors')==cp['weight_sha256']
print('ALL_CURRENT_WEIGHTS_HASH_VERIFIED',len(weights),flush=True)
ssdecoder=next(cp for prefix,cp in weights.items() if 'ss_dec_' in prefix);dec=models.from_pretrained(ssdecoder['prefix']);dec.convert_to_fp32();dec.float().cpu().eval();torch.set_num_threads(8)
for row in rows:
 with torch.inference_mode():logits=dec(torch.from_numpy(row.pop('mean')).unsqueeze(0));assert torch.isfinite(logits).all();coords=torch.argwhere(logits[0,0]>0).numpy()
 expected=row.pop('shared');saved=row.pop('saved_decoded');gs={tuple(p) for p in expected};ps={tuple(p) for p in coords};iou=len(gs&ps)/len(gs|ps);assert ps=={tuple(p) for p in saved};row['independent_cpu_ss_roundtrip_iou']=iou;row['ss_decoder_output_matches_saved']=True;print('INDEPENDENT_SS_DECODE',row['object_id'],iou,flush=True)
passed=all(r['independent_cpu_ss_roundtrip_iou']==1 and max(abs(x) for x in r['relative_extent_error'])<=.005 for r in rows)
report={'schema':'trellis.six-target-independent-validation.v1','passed':passed,'approved_for_short_training':passed,'scope':[r['object_id'] for r in rows],'condition_view':0,'production_approved':False,'numerical_contract_only':True,'shape_semantics_not_automatic_truth':True,'ss_support_equality_constructed_not_independent':True,'current_model_weight_hashes_verified':list(weights),'source_and_output_hashes_verified':True,'rows':rows,'extent_0_5_percent_reference_screen_not_universal_quality_threshold':True,'elapsed_seconds':time.perf_counter()-started,'render_metadata_not_recomputed':True}
(out/'validation.json').write_text(json.dumps(report,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'validation_sha256':sha(out/'validation.json'),'passed':passed,'production_approved':False},indent=2));print(json.dumps(report,indent=2))
