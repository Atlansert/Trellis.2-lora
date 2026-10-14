"""All-triangle CPU projection; fixed world axes, common unit scale. No mesh realignment."""
import os,sys,json,time,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,trimesh
from PIL import Image,ImageDraw
root=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(root));os.chdir(root)
from scripts.utils.evaluate_fixed_frame import world_mesh
b=root/'diagnostics/data_rebuild_batch_v1/formal_validation';report=json.loads((b/'v1_first_two/report.json').read_text());out=b/'v1_first_two/fixed_side_top_all_triangles.png'
if out.exists():raise FileExistsError(out)
W,H=420,245;S=350.;canvas=Image.new('RGB',(W*5,H*4+60),(24,27,32));d=ImageDraw.Draw(canvas)
d.text((10,8),'CPU complete-triangle union silhouettes, standard GLB world coordinates. Same unit length scale; no best view/rotation/ICP.',fill='white')
d.text((10,28),'Columns input / old side ZY / new side ZY / old top ZX / new top ZX. Grey shape is geometry, not textured rendering.',fill='white')
start=time.monotonic();records=[]
for row,n in enumerate(['001','002','003','004']):
 y=60+row*H;im=Image.open(root/'data_test'/f'{n}.png').convert('RGB');a=np.asarray(im);ys,xs=np.nonzero(a.max(2)>16);crop=im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1));crop.thumbnail((W-30,H-60));canvas.paste(crop,((W-crop.width)//2,y+(H-crop.height)//2));d.text((8,y+8),n+' input (no GT identity)',fill='white')
 for tag,cols in [('old33k',(1,3)),('rebuilt_ss_live50',(2,4))]:
  r=next(x for x in report['results'] if x['record']['id']==f'{tag}_{n}');path=Path(r['record']['glb'])
  if r['artifact_status']=='missing' or not path.is_file():
   for col in cols:d.text((col*W+8,y+8),f'{n} {tag} MISSING',fill=(255,160,130))
   continue
  mesh=world_mesh(trimesh.load(path,force='scene',process=False));v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);center=np.asarray(mesh.bounds).mean(0)
  for col,vertical,label in [(cols[0],1,'ZY side +Y up'),(cols[1],0,'ZX top +X up')]:
   tile=Image.new('RGB',(W,H),(31,35,41));draw=ImageDraw.Draw(tile);xy=np.rint(np.column_stack(((v[:,2]-center[2])*S+W/2,H/2-(v[:,vertical]-center[vertical])*S))).astype(int)
   # Every triangle included; integer PIL union includes boundary pixels and
   # can differ from antialias raster. No face sampling or area cutoff.
   for face in f:
    p=xy[face];draw.polygon([(int(x),int(z)) for x,z in p],fill=(184,194,203))
   m=r['metrics'];angle=m['surface_pca']['major_to_z_offangle_degrees'];draw.text((8,8),f'{n} {tag} {label}',fill='white');draw.text((8,H-35),f'long={m["longest_aabb_axis"]} offZ={angle:.3f}deg',fill='white');draw.text((8,H-19),f'z/y={m["z_over_y"]:.3f} x/y={m["x_over_y"]:.3f}',fill='white');canvas.paste(tile,(col*W,y))
  records.append({'id':r['record']['id'],'glb_sha256':r['artifact_sha256'],'world_vertices':len(v),'all_triangles':len(f),'world_instance_transform_applied':True})
  print('PROJECTED',n,tag,len(f),flush=True)
canvas.save(out);meta={'image':str(out),'image_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'method':'PIL all-triangle union fixedZY/ZX each unitlength350pixels; center each AABB translation only, no scale normalization/rotation/fit','records':records,'missing_ids':['rebuilt_ss_live50_003','rebuilt_ss_live50_004'],'elapsed_seconds':time.monotonic()-start,'production_approved':False};(out.with_suffix('.json')).write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
