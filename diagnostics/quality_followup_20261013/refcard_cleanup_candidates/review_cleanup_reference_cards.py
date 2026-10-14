import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,hashlib,time
import numpy as np,trimesh
from PIL import Image,ImageDraw
root=Path('/data/lzx/TRELLIS.2');os.chdir(root);out=root/'diagnostics/quality_followup_20261013/refcard_cleanup_candidates'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows={x['sha256']:x for x in csv.DictReader(open('training_data/multiview_1186/metadata.csv'))};records=[]
contact=Image.new('RGB',(1100,850),(30,33,37));d=ImageDraw.Draw(contact)
for ri,obj in enumerate(['0fab4af3f9a73bdd','f50d6a143bdfe293','5d4a5e8e46991642']):
 p=root/rows[obj]['local_path'];scene=trimesh.load(p,force='scene',process=False);rec={'object':obj,'source':str(p),'source_sha256':sha(p),'nodes':[]}
 for node in scene.graph.nodes_geometry:
  matrix,name=scene.graph[node];m=scene.geometry[name];material=getattr(m.visual,'material',None);texture=getattr(material,'baseColorTexture',None)
  if len(m.faces)!=2 or texture is None:continue
  im=texture.convert('RGBA');a=np.asarray(im.getchannel('A'));world=trimesh.transform_points(m.vertices,matrix)
  stem=f'{obj}_{node}'.replace('/','_');im.save(out/(stem+'.png'))
  record={'node':str(node),'geometry':str(name),'faces':len(m.faces),'material_name':getattr(material,'name',None),'alphaMode':getattr(material,'alphaMode',None),'texture_size':list(im.size),'pixel_sha256':hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),'texture_png_sha256':sha(out/(stem+'.png')),'alpha_values':np.unique(a).tolist(),'world_bounds':np.stack([world.min(0),world.max(0)]).tolist(),'atlas_visual_review':'pending'}
  rec['nodes'].append(record)
 if rec['nodes']:
  im=Image.open(out/(f"{obj}_{rec['nodes'][0]['node']}.png"));im.thumbnail((1000,220));contact.paste(im.convert('RGB'),(30,ri*275+40));d.text((20,ri*275+10),obj+' node '+rec['nodes'][0]['node']+' '+str(rec['nodes'][0]['texture_size']),fill='white')
 records.append(rec)
contact.save(out/'reference_atlas_contact.png')
# Exact whitelist proven by prior material/atlas read; no generic BLEND filtering.
p=root/rows['0fab4af3f9a73bdd']['local_path'];original_hash=sha(p);s=trimesh.load(p,force='scene',process=False);before=s.bounds.copy();removed=[]
for node in ['Fromt','Side']:
 matrix,name=s.graph[node];mesh=s.geometry[name];assert len(mesh.faces)==2 and mesh.visual.material.alphaMode=='BLEND';removed.append({'node':node,'geometry':name,'texture_pixels_sha256':hashlib.sha256(np.asarray(mesh.visual.material.baseColorTexture.convert('RGBA')).tobytes()).hexdigest()});s.graph.transforms.remove_node(node)
# Preserve every remaining geometry instance, world transform, visual/material.
clean=trimesh.Scene()
for node in s.graph.nodes_geometry:
 matrix,name=s.graph[node];clean.add_geometry(s.geometry[name].copy(),node_name=node,geom_name=name,transform=matrix)
assert '1435' in clean.graph.nodes_geometry
candidate=out/'0fab4af3f9a73bdd_remove_Fromt_Side_candidate.glb';assert not candidate.exists();candidate.write_bytes(trimesh.exchange.gltf.export_glb(clean));assert sha(p)==original_hash
reloaded=trimesh.load(candidate,force='scene',process=False)
cleanup={'source_sha256':original_hash,'candidate_sha256':sha(candidate),'removed_exact_whitelist':removed,'before_world_bounds':before.tolist(),'after_world_bounds':clean.bounds.tolist(),'remaining_geometry_instances':len(clean.graph.nodes_geometry),'reloaded_geometry_instances':len(reloaded.graph.nodes_geometry),'preserved_unknown_cube_node':'1435','original_source_unchanged':True,'material_preservation':'copied remaining scene visual/material per instance; GLB exporter re-embeds textures; exact serialized material fidelity not guaranteed','training_approved':False}
(out/'review.json').write_text(json.dumps({'assets':records,'cleanup_candidate':cleanup},indent=2));print(json.dumps(cleanup,indent=2))
# Same camera (+rawX) silhouettes at identical ORIGINAL normalized scale.
canvas=Image.new('RGB',(1000,320),(30,33,37));draw=ImageDraw.Draw(canvas)
source=trimesh.load(p,force='scene',process=False);bounds=source.bounds;center=bounds.mean(0);scale=.99999/np.ptp(bounds,axis=0).max()
for col,(label,scene) in enumerate([('Original all geometry',source),('Candidate: only Fromt/Side removed',reloaded)]):
 for node in scene.graph.nodes_geometry:
  matrix,name=scene.graph[node];mesh=scene.geometry[name];v=(trimesh.transform_points(mesh.vertices,matrix)-center)*scale
  # Native x-long asset yaw90 yields common rawY=-nativeX; rawZ=nativeY.
  xy=np.rint(np.stack([col*500+250-v[:,0]*450,170-v[:,1]*450],axis=1)).astype(int)
  for f in mesh.faces:draw.polygon([tuple(q) for q in xy[f]],fill=(180,185,190))
 draw.text((col*500+10,10),label,fill='white')
canvas.save(out/'cleanup_same_camera.png');print('DONE',out)
