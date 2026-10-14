"""CPU geometry-only hypothesis. No model construction, latent transpose or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
root=Path('/data/lzx/TRELLIS.2');b=root/'diagnostics/cohort_training_followup_v1';batch=root/'diagnostics/data_rebuild_batch_v1';out=b/'nose_canonical_candidate';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=b/'nose_contract_review/review.json';nose=json.loads(review.read_text());render=json.loads((batch/'batch_render_validation.json').read_text());pilot=json.loads((batch/'short_training_prepare/contract_v1/manifest.json').read_text());by={x['object_id']:x for x in pilot['objects']}
expected={'0086bbab37559442':-1,'008ae427b2606cc8':-1,'015de7a65b71fbc2':-1,'0390f705ac242a39':-1,'1208e4cc0c30448a':1,'1f7cc1ae1cec73f5':1}
records=[];start=time.monotonic();canvas=Image.new('RGB',(1500,6*190),(22,24,28));draw=ImageDraw.Draw(canvas)
for i,e in enumerate(render['objects']):
 obj=e['object_id'];r=next(x for x in nose['records'] if x['object_id']==obj);sign=r['geometry_taper']['candidate_nose_rawY_sign'];assert sign==expected[obj]
 old=Path(e['render_directory'])/'canonical_geometry.npz';oldhash=sha(old);assert oldhash==r['geometry_sha256'];d=out/obj;d.mkdir()
 with np.load(old,allow_pickle=False) as z:v=z['vertices'].copy();faces=z['faces'].copy()
 R=np.diag([-1.,-1.,1.]) if sign==1 else np.eye(3);new=(v@R).astype(v.dtype)
 assert abs(np.linalg.det(R)-1)<1e-12 and np.array_equal(new[:,2],v[:,2]);np.testing.assert_allclose(np.ptp(new,axis=0),np.ptp(v,axis=0),atol=1e-7)
 dst=None
 if sign==1:
  dst=d/'geometry_candidate.npz';np.savez_compressed(dst,vertices=new,faces=faces)
  with np.load(dst,allow_pickle=False) as z:assert np.array_equal(z['faces'],faces) and np.array_equal(z['vertices'],new)
  # Orientation of each sampled face transforms with the proper matrix, no winding changes.
  f=faces[::max(1,len(faces)//10000)];n1=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);n2=np.cross(new[f[:,1]]-new[f[:,0]],new[f[:,2]]-new[f[:,0]]);np.testing.assert_allclose(n2,n1@R,atol=1e-9,rtol=1e-6)
 assert sha(old)==oldhash
 condition=by[obj]['condition'];assert sha(condition['path'])==condition['sha256'];image=Image.open(condition['path']).convert('RGB');image.thumbnail((470,160));canvas.paste(image,(10,i*190+25));draw.text((10,i*190+4),obj+' unchanged source condition',fill='white')
 # Fixed camera+rawX/right+Y/up+Z. Vertex projection diagnostic, not filled silhouette.
 for col,(label,vertices) in enumerate([('old proper geometry',v),('candidate canonical nose -Y',new)],start=1):
  tile=Image.new('RGB',(480,155),(30,34,40));td=ImageDraw.Draw(tile)
  for pt in vertices[::max(1,len(vertices)//30000)]:
   x=int((pt[1]+.5)*440)+20;y=100-int(pt[2]*440);td.point((x,y),fill=(100,195,225))
  td.text((5,5),label,fill='white');canvas.paste(tile,(col*500,i*190+25))
 # Explicit camera extrinsics transform if same image is expressed in new canonical frame.
 rm=json.loads((Path(e['render_directory'])/'manifest.json').read_text());C=np.asarray(rm['views'][0]['matrix_world_camera_to_world']);T=np.eye(4);T[:3,:3]=R.T;newC=T@C
 rec={'object_id':obj,'split':by[obj]['split'],'source_geometry':{'path':str(old),'sha256':oldhash},'source_condition':condition,'condition_unchanged_hash_verified':True,'source_nose_rawY_candidate':sign,'canonical_nose_rawY_candidate':-1,'matrix_row_oldraw_to_canonical':R.tolist(),'determinant':1,'geometry_candidate':None if dst is None else {'path':str(dst),'sha256':sha(dst)},'identity_geometry_reference':str(old) if dst is None else None,'face_winding_unchanged':True,'up_preserved':True,'bbox_extents_preserved':True,'source_geometry_unchanged':True,'camera0_to_world_oldraw':C.tolist(),'same_image_camera_to_world_newcanonical':newC.tolist(),'new_shape_latent':None,'new_ss_mean':None,'training_approved':False,'semantic_review':'source images visually support fourleft/tworight; end taper is a program candidate, not measured keypoint; parent approval still required'}
 records.append(rec);print('CANDIDATE',obj,'rotate180',sign==1,flush=True)
canvas.save(out/'source_old_candidate_contact.png')
manifest={'schema':'trellis.nose-canonical-geometry-candidate.v1','production_approved':False,'training_approved':False,'geometry_only':True,'condition_images_changed':False,'model_predictions_changed':False,'target_definition':'pose-invariant canonical vehicle nose -rawY, height +rawZ, length rawY; source image remains selected physical view','required_before_training':['parent semantic source-anchor review','actual shape VAE reencoding from new geometry','SS encoding from same new shape64support','roundtrip numerical and semantic checks','new hashbound scope approval'],'controls':'Only target frame definition changes; do not combine with duration experiment; start from same official initialization/recipe/split','heldout_policy':'rotation chosen from source geometry/images, never predicted output or best IoU','encoded_latent_permutation_forbidden':True,'formal_acceptance':'raw output evaluated directly in this newly declared target frame; no posthoc correction','review_sha256':sha(review),'script_sha256':sha(__file__),'records':records,'elapsed_seconds':time.monotonic()-start}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha(out/'manifest.json'),'contact_sha256':sha(out/'source_old_candidate_contact.png'),'geometry_candidates':2,'identity_references':4,'training_approved':False},indent=2));print('GEOMETRY_ONLY_CANDIDATE_COMPLETE',manifest['elapsed_seconds'],flush=True)
