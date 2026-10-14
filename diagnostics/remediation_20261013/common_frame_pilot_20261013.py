import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
import sys,time,json,hashlib,csv
from pathlib import Path
import numpy as np
import trimesh,open3d as o3d
ROOT=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(ROOT))
from data_toolkit.canonical_frame import native_gltf_to_internal,internal_to_formal_glb,FRAME_VERSION
OUT=ROOT/'diagnostics/remediation_20261013/common_frame_pilot';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def keys(c):return set(map(tuple,c.tolist()))
def iou(a,b):a=keys(a);b=keys(b);return len(a&b)/len(a|b)
rows={x['sha256']:x for x in csv.DictReader((ROOT/'training_data/multiview_1186/metadata.csv').open())};reports=[]
for obj,yaw in [('0086bbab37559442',0.),('0fab4af3f9a73bdd',90.)]:
 src=ROOT/rows[obj]['local_path'];mesh=trimesh.load(src,force='mesh');v=np.asarray(mesh.vertices);v=(v-(v.min(0)+v.max(0))/2)*(.99999/np.ptp(v,axis=0).max());raw=native_gltf_to_internal(v,yaw);formal=internal_to_formal_glb(raw)
 p=OUT/obj;p.mkdir();np.savez_compressed(p/'proper_geometry.npz',vertices=raw.astype('float32'),faces=np.asarray(mesh.faces).astype('int32'))
 mm=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(raw),o3d.utility.Vector3iVector(np.asarray(mesh.faces)));grid=o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mm,1/64,(-.5,-.5,-.5),(.5,.5,.5));c=np.asarray([x.grid_index for x in grid.get_voxels()],dtype=np.int32);c=np.unique(np.clip(c,0,63),axis=0)
 # Identical support by construction, not a shape VAE encoding.
 np.savez_compressed(p/'ss_support64.npz',coords=c);np.savez_compressed(p/'shape_support64_geometry_only.npz',coords=c.copy())
 old=ROOT/'training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024'/f'{obj}.npz';oc=np.load(old)['coords'];maxc=int(oc.max());gridres=64 if maxc<64 else 1024
 if gridres!=64:oc=np.unique(oc//(gridres//64),axis=0)
 centers=(oc.astype(float)+.5)/64-.5;newold=np.floor((native_gltf_to_internal(centers,yaw)+.5)*64+1e-8).astype(int);newold=np.unique(np.clip(newold,0,63),axis=0)
 marks=native_gltf_to_internal(np.array([[0,1,0],[0,-1,0],[1,0,0],[0,0,1]],float),yaw)
 mat=native_gltf_to_internal(np.eye(3),yaw)
 report={'object':obj,'source':str(src),'source_sha256':sha(src),'yaw':yaw,'matrix_row':mat.tolist(),'determinant':float(np.linalg.det(mat)),'native_up_label':[0,1,0],'raw_up_marker':marks[0].tolist(),'formal_up_marker':internal_to_formal_glb(marks)[0].tolist(),'raw_spans':np.ptp(raw,axis=0).tolist(),'formal_spans':np.ptp(formal,axis=0).tolist(),'support_count':len(c),'support_shared_equal':True,'support_pair_equality_reason':'Constructed from same mesh support, NOT independently encoded shape VAE','old_shape_coords_path':str(old),'old_shape_sha256':sha(old),'old_shape_grid_inferred':gridres,'old_transformed_support_vs_new_geometry_iou':iou(newold,c),'geometry_sha256':sha(p/'proper_geometry.npz'),'ss_support_sha256':sha(p/'ss_support64.npz')}
 reports.append(report);print('GEOMETRY',json.dumps(report),flush=True)
encbase=Path('/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16');decbase=Path('/home/test/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16')
manifest={'frame_version':FRAME_VERSION,'training_approved':False,'script_sha256':sha(__file__),'helper_sha256':sha(ROOT/'data_toolkit/canonical_frame.py'),'assets':reports,'encoder':{'base':str(encbase),'json':json.loads(encbase.with_suffix('.json').read_text()),'weights_sha256':sha(encbase.with_suffix('.safetensors'))},'decoder':{'base':str(decbase),'json':json.loads(decbase.with_suffix('.json').read_text()),'weights_sha256':sha(decbase.with_suffix('.safetensors'))},'limitations':['No shape/PBR VAE encoding or decoded shape validation','Fixed shape20k compatibility unproven','No image-camera/nose semantic correlation validation','Native Y-up marker is coordinate convention label not ground-truth roof recognition']};atom(OUT/'manifest.json',manifest)
import torch
import trellis2.models as models
torch.set_num_threads(8);torch.set_num_interop_threads(2);started=time.monotonic()
enc=models.from_pretrained(str(encbase)).eval();enc.convert_to_fp32();enc.float().cpu();dec=models.from_pretrained(str(decbase)).eval();dec.convert_to_fp32();dec.float().cpu()
for r in reports:
 if time.monotonic()-started>180:print('CPU_ENCODE_BUDGET_STOP',flush=True);break
 p=OUT/r['object'];c=np.load(p/'ss_support64.npz')['coords'];x=torch.zeros((1,1,64,64,64),dtype=torch.float32);x[0,0,c[:,0],c[:,1],c[:,2]]=1
 with torch.inference_mode():z=enc(x,sample_posterior=False);d=dec(z);occ=d>0
 assert tuple(z.shape)==(1,8,16,16,16) and torch.isfinite(z).all() and torch.isfinite(d).all();pred=occ[0,0].nonzero().numpy()
 np.savez_compressed(p/'new_ss_mean.npz',mean=z[0].numpy());np.savez_compressed(p/'decoded_ss_support64.npz',coords=pred)
 r['ss_vae']={'shape':list(z[0].shape),'dtype':str(z.dtype),'finite':True,'decoded_iou':iou(c,pred),'decoded_count':len(pred),'elapsed_total_seconds':time.monotonic()-started,'mean_sha256':sha(p/'new_ss_mean.npz')};print('VAE',r['object'],r['ss_vae'],flush=True);atom(OUT/'manifest.json',manifest)
atom(OUT/'COMPLETE.json',{'geometry_assets':len(reports),'ss_vae_assets':sum('ss_vae'in r for r in reports),'manifest_sha256':sha(OUT/'manifest.json'),'training_approved':False});print('PILOT_COMPLETE',flush=True)
