"""Blender4.5 reviewed-source ONLY common-frame render; preparation until parent executes.
Usage: blender --background --factory-startup --python render_common_frame.py --
       --review /abs/review.json --output /abs/NEW --device cpu
No renderer execution is performed merely by importing the CPU contract/tests.
"""
import sys,os,json,time,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from render_common_contract import (sha,atomic_json,validate_review,rz_row,native_to_raw_matrix,
    normalization,camera_to_world,intrinsic,project_world,validate_alpha,fresh_output,strict_cuda_environment,RAW_AXES)

def world_geometry(bpy,objects):
 deps=bpy.context.evaluated_depsgraph_get();vertices=[];faces=[];offset=0;records=[]
 for obj in objects:
  evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh()
  try:
   mesh.calc_loop_triangles();v=np.empty((len(mesh.vertices),3),np.float64);mesh.vertices.foreach_get('co',v.ravel());matrix=np.asarray(evaluated.matrix_world,dtype=np.float64)
   if np.linalg.det(matrix[:3,:3])<=0:raise ValueError(f'negative/degenerate world transform {obj.name}; needs separate review, not automatic winding repair')
   v=v@matrix[:3,:3].T+matrix[:3,3];f=np.empty((len(mesh.loop_triangles),3),np.int32);mesh.loop_triangles.foreach_get('vertices',f.ravel())
   if not len(v) or not len(f):raise ValueError(f'empty imported mesh {obj.name}')
   vertices.append(v);faces.append(f.astype(np.int64)+offset);offset+=len(v)
   records.append({'name':obj.name,'mesh_data':obj.data.name,'vertices':len(v),'triangles':len(f),'matrix_world':matrix.tolist(),'bounds':[v.min(0).tolist(),v.max(0).tolist()],
                   'material_slots':[slot.material.name if slot.material else None for slot in obj.material_slots]})
  finally:evaluated.to_mesh_clear()
 return np.concatenate(vertices),np.concatenate(faces),records

def main():
 import argparse,bpy
 from mathutils import Matrix
 argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
 ap=argparse.ArgumentParser();ap.add_argument('--review',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--threads',type=int,default=8);ap.add_argument('--resolution',type=int,default=512);ap.add_argument('--samples',type=int,default=16);ap.add_argument('--radius',type=float,default=2.2);ap.add_argument('--fov',type=float,default=30)
 args=ap.parse_args(argv)
 if args.resolution!=512 or args.samples!=16 or args.threads!=8:raise ValueError('batch-v1 contract is actual512/samples16/CPUthreads8; create reviewed version for other options')
 if args.device=='cuda':strict_cuda_environment(os.environ.get('CUDA_VISIBLE_DEVICES'))
 if bpy.app.version[:2]!=(4,5):raise ValueError('reviewed renderer requires Blender4.5')
 review,source,native_nodes=validate_review(args.review)
 out=fresh_output(args.output);manifest={'schema':'trellis.common-frame-render.v1','status':'initializing','views':[],'training_approved':False,'source_sha256':sha(source),'source_path':str(source),'review_sha256':sha(args.review),'review_path':str(Path(args.review).resolve()),'script_sha256':sha(__file__),'contract_sha256':sha(Path(__file__).with_name('render_common_contract.py')),'blender_version':bpy.app.version_string,'options':vars(args),'raw_axes':RAW_AXES,'selected_yaw_degrees':review['selected_yaw_degrees'],'native_to_raw_matrix_row':native_to_raw_matrix(review['selected_yaw_degrees']).tolist(),'native_nodes_reviewed':native_nodes,'nose_direction':'preserve reviewed selected side; no semantic nose flip; initial conditional training view0 only','formal_rotation_row':[[1,0,0],[0,0,-1],[0,1,0]],'camera_convention':'Blender camera x-right/y-up/looks local -Z; CV pixels x-right/y-down; principal point uses pixel-edge coordinates width/2,height/2','scene_nodes_removed':[]}
 def save():atomic_json(out/'manifest.json',manifest)
 save();started=time.perf_counter()
 try:
  for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
  if bpy.app.debug_value==100:raise ValueError('Blender debug100 disables native glTF axis conversion')
  bpy.ops.import_scene.gltf(filepath=str(source))
  imported=list(bpy.context.scene.objects);meshes=[o for o in imported if o.type=='MESH']
  if not meshes:raise ValueError('imported scene has no meshes')
  # All source meshes are retained. No name-based deletion or automatic cleanup.
  if any(o.hide_render for o in meshes):raise ValueError('hidden imported geometry requires separate asset review')
  exact=review.get('expected_blender_mesh_object_names')
  if exact is not None and sorted(exact)!=sorted(o.name for o in meshes):raise ValueError('actual imported mesh names differ from explicit review')
  expected_count=review.get('expected_blender_mesh_object_count')
  if expected_count is not None and expected_count!=len(meshes):raise ValueError('actual imported mesh count differs from explicit review')
  roots=[o for o in imported if o.parent is None];common=bpy.data.objects.new('TRELLIS_COMMON_FRAME_ROOT',None);bpy.context.scene.collection.objects.link(common)
  for obj in roots:
   old=obj.matrix_world.copy();obj.parent=common;obj.matrix_world=old
  # Importer ALREADY converts native glTF (x,y,z)->Blender(x,-z,y).
  # Row Rz(yaw) is the transpose/column Rz(-yaw), never duplicate import mapping.
  yaw_matrix=np.eye(4);yaw_matrix[:3,:3]=rz_row(review['selected_yaw_degrees']).T;common.matrix_world=Matrix(yaw_matrix.tolist());bpy.context.view_layer.update()
  before_v,_,before_records=world_geometry(bpy,meshes);center,scale,N=normalization(before_v);common.matrix_world=Matrix((N@yaw_matrix).tolist());bpy.context.view_layer.update()
  v,f,actual=world_geometry(bpy,meshes)
  expected=(before_v-center)*scale
  if not np.allclose(v,expected,rtol=1e-6,atol=2e-6):raise ValueError('hierarchy/common transform differs from recorded normalization')
  if np.argmax(np.ptp(v,axis=0))!=1:raise ValueError('review selected yaw did not produce common raw lengthY')
  gp=out/'canonical_geometry.npz';np.savez_compressed(gp,vertices=v.astype(np.float32),faces=f.astype(np.int32))
  manifest.update(geometry={'path':gp.name,'sha256':sha(gp),'vertices':len(v),'faces':len(f),'bounds':[v.min(0).tolist(),v.max(0).tolist()], 'world_space':'common raw depthX/lengthY/upZ; unchanged for all cameras'},normalization={'center_after_yaw':center.tolist(),'scale':scale,'fit':0.99999,'matrix_column':N.tolist(),'yaw_matrix_column':yaw_matrix.tolist(),'root_world_matrix_column':(N@yaw_matrix).tolist()},actual_imported_objects=[{'name':o.name,'type':o.type,'parent':o.parent.name if o.parent else None} for o in imported],actual_meshes=actual,all_reviewed_source_meshes_retained=True)
  scene=bpy.context.scene;scene.render.engine='CYCLES';scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8';scene.render.film_transparent=True;scene.render.threads_mode='FIXED';scene.render.threads=8
  scene.cycles.samples=16;scene.cycles.use_denoising=True;scene.cycles.filter_type='BOX';scene.cycles.filter_width=1
  if args.device=='cpu':scene.cycles.device='CPU'
  else:
   prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='CUDA';prefs.get_devices();cuda=[d for d in prefs.devices if d.type=='CUDA']
   if len(cuda)!=1:raise ValueError('exactly one visible CUDA device required; no automatic multiGPU choice')
   for d in prefs.devices:d.use=(d.type=='CUDA')
   scene.cycles.device='GPU';manifest['cuda_devices']=[{'name':d.name,'type':d.type,'use':bool(d.use)} for d in prefs.devices]
  scene.display_settings.display_device='sRGB';scene.view_settings.view_transform='Filmic';scene.view_settings.look='None';scene.view_settings.exposure=0.;scene.view_settings.gamma=1.
  # Reviewed batch renders use one controlled white environment. Imported lights
  # are disabled/recorded (not deleted); mesh/materials are retained unchanged.
  imported_lights=[o for o in imported if o.type=='LIGHT']
  for o in imported_lights:o.hide_render=True
  manifest['disabled_imported_lights']=[o.name for o in imported_lights]
  world=bpy.data.worlds.new('TRELLIS_WHITE_ENV');scene.world=world;world.use_nodes=True;bg=world.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(1,1,1,1);bg.inputs['Strength'].default_value=1
  camera=bpy.data.objects.new('TRELLIS_RENDER_CAMERA',bpy.data.cameras.new('TRELLIS_RENDER_CAMERA'));scene.collection.objects.link(camera);scene.camera=camera;camera.data.type='PERSP';camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=32;camera.data.shift_x=camera.data.shift_y=0;camera.data.clip_start=.01;camera.data.clip_end=100
  K,lens=intrinsic(512,512,args.fov,32);camera.data.lens=lens
  manifest.update(status='rendering',color_management={'view_transform':'Filmic','look':'None','exposure':0,'gamma':1,'display':'sRGB','environment':'white1; transparent background','same_as_historical_pixels_verified':False},camera_intrinsic={'K_cv':K.tolist(),'lens_mm':lens,'sensor_width_mm':32,'sensor_fit':'HORIZONTAL','fov_requested_degrees':args.fov,'angle_x_actual_radians':camera.data.angle_x,'pixel_aspect':[1,1]})
  save()
  # Reprojection audit points use geometry actually exported, not source-local.
  sample=v[np.linspace(0,len(v)-1,min(32,len(v)),dtype=int)]
  for index,yaw in enumerate(range(0,360,45)):
   t=time.perf_counter();C=camera_to_world(yaw,args.radius);camera.matrix_world=Matrix(C.tolist());bpy.context.view_layer.update();actualC=np.asarray(camera.matrix_world);W=np.linalg.inv(actualC);pixels,depth=project_world(sample,W,K)
   image=out/f'{index:03d}.png'
   if image.exists():raise FileExistsError(image)
   scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
   # Inspect actual saved pixel alpha via Blender image loader, no segmentation.
   saved=bpy.data.images.load(str(image),check_existing=False)
   try:
    pixels_rgba=np.empty(saved.size[0]*saved.size[1]*4,np.float32);saved.pixels.foreach_get(pixels_rgba);rgba=np.rint(np.clip(pixels_rgba.reshape(saved.size[1],saved.size[0],4)[::-1],0,1)*255).astype(np.uint8);quality=validate_alpha(rgba)
   finally:bpy.data.images.remove(saved)
   manifest['views'].append({'index':index,'camera_yaw_degrees':yaw,'file':image.name,'sha256':sha(image),'elapsed_seconds':time.perf_counter()-t,'matrix_world_camera_to_world':actualC.tolist(),'matrix_world_to_camera':W.tolist(),'intrinsic_K_cv':K.tolist(),'camera_position_world':actualC[:3,3].tolist(),'camera_right_world':actualC[:3,0].tolist(),'camera_up_world':actualC[:3,1].tolist(),'camera_forward_world':(-actualC[:3,2]).tolist(),'projection_audit_world_points':sample.tolist(),'projection_audit_pixels_cv':pixels.tolist(),'projection_depths':depth.tolist(),'alpha':quality})
   save();print('VIEW_COMPLETE',index,yaw,quality,flush=True)
  if sha(source)!=manifest['source_sha256'] or sha(args.review)!=manifest['review_sha256']:raise ValueError('source/review changed during render')
  manifest.update(status='success',elapsed_seconds=time.perf_counter()-started);save();atomic_json(out/'COMPLETE.json',{'schema':'trellis.render-complete.v1','views':8,'manifest_sha256':sha(out/'manifest.json'),'training_approved':False});print('RENDER_COMPLETE_ALL8_NOT_TRAINING_APPROVAL',flush=True)
 except Exception as e:
  manifest.update(status='failed',error=f'{type(e).__name__}: {e}',elapsed_seconds=time.perf_counter()-started);save();raise

if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();sys.exit(1)
