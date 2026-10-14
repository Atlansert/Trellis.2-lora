"""CPU-only contract/math for a reviewed glTF Y-up -> common raw Z-up render.
Blender import performs (x,y,z)->(x,-z,y). Apply ONLY world yaw afterward.
This module never imports bpy and never starts a render.
"""
import hashlib,json,math,os,struct
from pathlib import Path
import numpy as np
SCHEMA='trellis.asset-render-review.v1'
RAW_AXES={'depth':'x','length':'y','up':'z','camera0_position_axis':'+x','camera0_screen_right':'+y'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path,value):
 p=Path(path);tmp=p.with_name(p.name+'.tmp')
 if tmp.exists():raise FileExistsError(tmp)
 with open(tmp,'x',encoding='utf-8') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def glb_json(path):
 with open(path,'rb') as f:
  header=f.read(12)
  if len(header)!=12:raise ValueError('truncated GLB header')
  magic,version,length=struct.unpack('<III',header)
  if magic!=0x46546c67 or version!=2 or length!=Path(path).stat().st_size:raise ValueError('invalid GLB v2 header')
  while f.tell()<length:
   h=f.read(8)
   if len(h)!=8:raise ValueError('truncated chunk')
   n,kind=struct.unpack('<II',h)
   if n>length-f.tell():raise ValueError('chunk exceeds GLB')
   if kind==0x4e4f534a:return json.loads(f.read(n).decode('utf-8'))
   f.seek(n,1)
 raise ValueError('GLB JSON missing')

def mesh_nodes(document):
 nodes=document.get('nodes',[]);meshes=document.get('meshes',[])
 result=[]
 for i,n in enumerate(nodes):
  if 'mesh' in n:
   index=n['mesh']
   if type(index)!=int or not 0<=index<len(meshes):raise ValueError('invalid native mesh index')
   result.append({'node_index':i,'name':n.get('name',''),'mesh_index':index})
 if not result:raise ValueError('no reviewed source mesh nodes')
 return result

def validate_review(path):
 p=Path(path).resolve();review=json.loads(p.read_text(encoding='utf-8'))
 if review.get('schema')!=SCHEMA or review.get('review_passed') is not True:raise ValueError('asset review not passed')
 if review.get('retain_all_reviewed_meshes') is not True:raise ValueError('renderer never deletes source nodes; use separately reviewed cleaned GLB')
 src=Path(review.get('source',{}).get('path',''))
 if not src.is_absolute() or '..' in src.parts or src.suffix.lower()!='.glb' or not src.is_file():raise ValueError('absolute complete local GLB required')
 if sha(src)!=review['source'].get('sha256'):raise ValueError('review source hash changed')
 document=glb_json(src)
 for group in ('buffers','images'):
  for item in document.get(group,[]):
   uri=item.get('uri','')
   if uri and not uri.startswith('data:'):raise ValueError('external glTF buffers/textures require separately bound dependency review; embedded GLB only in batch-v1')
 native=mesh_nodes(document)
 if 'reviewed_native_mesh_nodes' in review:
  if review['reviewed_native_mesh_nodes']!=native:raise ValueError('native mesh-node whitelist differs from reviewed GLB')
 else:
  # Parent asset-review schema binds source bytes and exact native topology;
  # do not mutate/re-issue the parent's approval merely to rename keys.
  if review.get('approved_for_render') is not True:raise ValueError('asset review render approval missing')
  if review.get('approved_source_mesh_node_names')!=[x['name'] for x in native]:raise ValueError('native mesh-node whitelist differs from reviewed GLB')
  topology=review.get('source_scene_topology',{}).get('nodes',[])
  expected=[{'node_index':x['index'],'name':x.get('name',''),'mesh_index':x['mesh']} for x in topology if x.get('mesh') is not None]
  if expected!=native:raise ValueError('reviewed native node indices/mesh references differ')
 if review.get('source_coordinate_convention')!='gltf_y_up' and 'matrix_row_native_to_raw' not in review:raise ValueError('explicit glTF Y-up convention review required')
 yaw=review.get('selected_yaw_degrees')
 if not isinstance(yaw,(int,float)) or isinstance(yaw,bool) or not math.isfinite(yaw):raise ValueError('finite selected yaw required')
 if not review.get('object_id'):raise ValueError('review identity missing')
 if not ((review.get('reviewer') and review.get('review_evidence')) or (review.get('review_scope') and review.get('manual_image_observation'))):raise ValueError('review identity/evidence missing')
 for key in ('native_to_raw_matrix_row','matrix_row_native_to_raw'):
  if key in review:np.testing.assert_allclose(np.asarray(review[key]),native_to_raw_matrix(yaw),atol=1e-10,rtol=0)
 return review,src,native

def rz_row(degrees):
 if not math.isfinite(degrees):raise ValueError('finite yaw required')
 a=math.radians(degrees);c,s=math.cos(a),math.sin(a)
 return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)

def native_to_raw_matrix(yaw):
 # Row native Y-up->Blender Z-up, then row Rz(yaw).
 return np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=np.float64)@rz_row(yaw)

def normalization(vertices,fit=0.99999):
 v=np.asarray(vertices,dtype=np.float64)
 if v.ndim!=2 or v.shape[1]!=3 or len(v)<3 or not np.isfinite(v).all():raise ValueError('finite nonempty Nx3 vertices required')
 if not 0<fit<=1:raise ValueError('fit must be 0..1')
 lo=v.min(0);hi=v.max(0);extent=hi-lo
 if extent.max()<=0:raise ValueError('degenerate mesh')
 center=(lo+hi)/2;scale=float(fit/extent.max());matrix=np.eye(4);matrix[:3,:3]*=scale;matrix[:3,3]=-center*scale
 return center,scale,matrix

def camera_to_world(yaw,radius=2.2):
 if radius<=0 or not math.isfinite(radius):raise ValueError('positive finite camera radius required')
 a=math.radians(yaw);pos=radius*np.array([math.cos(a),math.sin(a),0.]);forward=-pos/radius;up=np.array([0.,0.,1.]);right=np.cross(forward,up);right/=np.linalg.norm(right);camup=np.cross(right,forward)
 matrix=np.eye(4);matrix[:3,:3]=np.column_stack([right,camup,-forward]);matrix[:3,3]=pos
 return matrix

def intrinsic(width=512,height=512,fov_degrees=30,sensor_width_mm=32):
 if width<=0 or height<=0 or not 0<fov_degrees<180 or sensor_width_mm<=0:raise ValueError('invalid camera settings')
 lens=sensor_width_mm/(2*math.tan(math.radians(fov_degrees)/2));fx=width*lens/sensor_width_mm
 return np.array([[fx,0,width/2],[0,fx,height/2],[0,0,1]],dtype=np.float64),lens

def project_world(vertices,world_to_camera,K):
 v=np.asarray(vertices);p=np.column_stack([v,np.ones(len(v))])@np.asarray(world_to_camera).T;depth=-p[:,2]
 if (depth<=0).any():raise ValueError('points behind camera')
 pixels=np.column_stack([K[0,0]*p[:,0]/depth+K[0,2],K[1,2]-K[1,1]*p[:,1]/depth])
 return pixels,depth

def validate_alpha(rgba):
 a=np.asarray(rgba)
 if a.ndim!=3 or a.shape[2]!=4 or a.dtype!=np.uint8:raise ValueError('uint8 RGBA required')
 alpha=a[:,:,3]
 if not np.any(alpha>204):raise ValueError('empty/no strong render alpha')
 if np.all(alpha==255):raise ValueError('opaque full-frame render alpha')
 y,x=np.nonzero(alpha>204)
 return {'coverage_strong':float((alpha>204).mean()),'coverage_nonzero':float((alpha>0).mean()),'bbox_strong_xyxy':[int(x.min()),int(y.min()),int(x.max()),int(y.max())],'mask_method':'direct rendered alpha; no RGB background derivation'}

def fresh_output(path):
 p=Path(path).resolve()
 if p.exists():raise FileExistsError('render output exists; partial artifacts are preserved; select new directory')
 p.mkdir(parents=True);return p

def strict_cuda_environment(value):
 if value not in ('6','7'):raise ValueError('CUDA render requires exactly CUDA_VISIBLE_DEVICES=6 or7')
 return value
