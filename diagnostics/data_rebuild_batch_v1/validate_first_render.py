import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
root=Path('/data/lzx/TRELLIS.2');src=root/'diagnostics/data_rebuild_batch_v1/renders/0086bbab37559442_cpu_v1';out=root/'diagnostics/data_rebuild_batch_v1/first_render_validation_v2';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();m=json.loads((src/'manifest.json').read_text());complete=json.loads((src/'COMPLETE.json').read_text());assert sha(src/'manifest.json')==complete['manifest_sha256'];assert m['status']=='success' and len(m['views'])==8
assert sha(m['source_path'])==m['source_sha256'] and sha(m['review_path'])==m['review_sha256']
assert sha(src/m['geometry']['path'])==m['geometry']['sha256']
with np.load(src/m['geometry']['path'],allow_pickle=False) as z:v=z['vertices'].astype(float);f=z['faces']
assert np.isfinite(v).all() and np.argmax(np.ptp(v,axis=0))==1
K=np.array([[512/(2*np.tan(np.deg2rad(30)/2)),0,256],[0,512/(2*np.tan(np.deg2rad(30)/2)),256],[0,0,1.]])
assert np.allclose(K,m['camera_intrinsic']['K_cv'],atol=1e-8)
assert abs(np.rad2deg(m['camera_intrinsic']['angle_x_actual_radians'])-30)<1e-4
native=np.array(m['native_to_raw_matrix_row']);assert np.allclose(native@native.T,np.eye(3),atol=1e-9) and abs(np.linalg.det(native)-1)<1e-9
assert np.allclose(np.array([0,1,0])@native,[0,0,1],atol=1e-9)
rows=[];contact=Image.new('RGB',(2048,1024),(45,45,45));draw=ImageDraw.Draw(contact)
for i,view in enumerate(m['views']):
 C=np.array(view['matrix_world_camera_to_world']);W=np.array(view['matrix_world_to_camera']);R=C[:3,:3]
 assert np.allclose(C@W,np.eye(4),atol=1e-8);assert np.allclose(R.T@R,np.eye(3),atol=1e-6) and abs(np.linalg.det(R)-1)<1e-6
 yaw=np.deg2rad(view['camera_yaw_degrees']);pos=2.2*np.array([np.cos(yaw),np.sin(yaw),0]);forward=-pos/2.2;right=np.cross(forward,[0,0,1]);up=np.cross(right,forward)
 assert np.allclose(C[:3,3],pos,atol=2e-7) and np.allclose(R[:,0],right,atol=1e-7) and np.allclose(R[:,1],up,atol=1e-7)
 points=np.array(view['projection_audit_world_points']);cam=np.column_stack([points,np.ones(len(points))])@W.T;depth=-cam[:,2]
 xy=np.column_stack([K[0,0]*cam[:,0]/depth+256,256-K[1,1]*cam[:,1]/depth]);error=float(np.max(np.abs(xy-np.array(view['projection_audit_pixels_cv']))));assert error<1e-7 and np.allclose(depth,view['projection_depths'],atol=1e-8)
 p=src/view['file'];assert sha(p)==view['sha256'];im=Image.open(p);assert im.mode=='RGBA' and im.size==(512,512);rgba=np.array(im);a=rgba[:,:,3];mask=a>204;ys,xs=np.nonzero(mask)
 assert mask.any() and not (a==255).all();bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())];assert bbox==view['alpha']['bbox_strong_xyxy'];assert min(xs.min(),ys.min())>0 and max(xs.max(),ys.max())<511
 cam=np.column_stack([v,np.ones(len(v))])@W.T;d=-cam[:,2];assert d.min()>.01;project=np.column_stack([K[0,0]*cam[:,0]/d+256,256-K[1,1]*cam[:,1]/d]);geom_bbox=[float(project[:,0].min()),float(project[:,1].min()),float(project[:,0].max()),float(project[:,1].max())]
 row={'view':i,'yaw':view['camera_yaw_degrees'],'alpha_coverage_nonzero':float((a>0).mean()),'alpha_coverage_strong':float(mask.mean()),'alpha_unique_count':len(np.unique(a)),'bbox_strong':bbox,'projected_geometry_bbox':geom_bbox,'reprojection_max_error_pixels':error,'edge_clipped':False}
 if i in (0,4):
  raster=Image.new('1',(512,512));rd=ImageDraw.Draw(raster)
  # Pixel centers are at edge coordinates x+.5/y+.5. Pillow is a binary,
  # inclusive triangle rasterizer, not Cycles subpixel coverage.
  coords=project-.5
  for tri in f:
   q=coords[tri];rd.polygon([tuple(x) for x in q],fill=1)
  rm=np.array(raster,dtype=bool);inter=(rm&mask).sum();union=(rm|mask).sum();row['binary_full_triangle_alpha_iou']=float(inter/union);row['alpha_outside_raster_pixels']=int((mask&~rm).sum());row['raster_outside_alpha_pixels']=int((rm&~mask).sum());raster.save(out/f'{i:03d}_geometry_silhouette.png');assert row['binary_full_triangle_alpha_iou']>.93
 checker=np.zeros((512,512,3),np.uint8);grid=np.indices((512,512)).sum(0)//16 # only display checker, not image inspection downsampling
 checker[:]=np.where((((np.indices((512,512))[0]//16)+(np.indices((512,512))[1]//16))%2)[...,None],170,220)
 tile=Image.fromarray(checker);tile.paste(im,mask=im.getchannel('A'));contact.paste(tile,((i%4)*512,(i//4)*512));ImageDraw.Draw(contact).text(((i%4)*512+8,(i//4)*512+8),f'{i} yaw{view["camera_yaw_degrees"]}',fill='black');rows.append(row);print('VALIDATED_VIEW',json.dumps(row),flush=True)
contact.save(out/'rgba_checker_all8.png')
result={'schema':'trellis.first-render-validation.v1','object':'0086bbab37559442','passed':True,'scope':'first asset transparent render and camera/geometry consistency only','training_approved':False,'all8_original_RGBA_sha_and_nonempty_nonopaque_unclipped_alpha_pass':True,'camera0_screen_right_world':[0,1,0],'camera0_up_world':[0,0,1],'independent_camera_K_matrix_reprojection_pass':True,'full_triangle_silhouette_test_views':[0,4],'silhouette_method_limits':'Pillow inclusive binary raster versus Cycles anti-aliasing/subpixel alpha; not exact pixel ground truth; no material cutout assumption generalized','geometry_extents':np.ptp(v,axis=0).tolist(),'views':rows,'render_manifest_sha256':sha(src/'manifest.json'),'script_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-start,'historical_RGB_lighting_equivalence_claimed':False,'manual_image_review':'pending parent/subagent direct image read'}
(out/'first_render_validation.json').write_text(json.dumps(result,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'report_sha256':sha(out/'first_render_validation.json'),'image_sha256':sha(out/'rgba_checker_all8.png'),'training_approved':False},indent=2));print('VALIDATION_COMPLETE',time.monotonic()-start,flush=True)
