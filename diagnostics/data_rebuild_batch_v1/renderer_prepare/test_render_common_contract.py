import unittest,tempfile,json,struct,math
from pathlib import Path
import numpy as np
from render_common_contract import *

def fixture(td):
 p=Path(td)/'source.glb';document={'asset':{'version':'2.0'},'nodes':[{'name':'Vehicle','mesh':0}],'meshes':[{'primitives':[]}],'scenes':[{'nodes':[0]}]};raw=json.dumps(document).encode();raw+=b' '*((-len(raw))%4);p.write_bytes(struct.pack('<III',0x46546c67,2,20+len(raw))+struct.pack('<II',len(raw),0x4e4f534a)+raw)
 review={'schema':SCHEMA,'review_passed':True,'retain_all_reviewed_meshes':True,'source':{'path':str(p),'sha256':sha(p)},'reviewed_native_mesh_nodes':mesh_nodes(document),'source_coordinate_convention':'gltf_y_up','selected_yaw_degrees':0,'object_id':'unit','reviewer':'CPU fixture only','review_evidence':'fixture; no real asset approved'};q=Path(td)/'review.json';q.write_text(json.dumps(review));return p,q,review

class ContractTests(unittest.TestCase):
 def test_proper_native_markers(self):
  r=native_to_raw_matrix(0);np.testing.assert_allclose(np.eye(3)@r,[[1,0,0],[0,0,1],[0,-1,0]]);self.assertAlmostEqual(np.linalg.det(r),1)
 def test_yaw_all_views_preserves_up_handedness(self):
  for yaw in range(0,360,45):
   r=native_to_raw_matrix(yaw);np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12);np.testing.assert_allclose(np.array([0,1,0])@r,[0,0,1],atol=1e-12);self.assertAlmostEqual(np.linalg.det(r),1)
 def test_row_yaw_equivalent_blender_column_negative_yaw(self):
  v=np.array([[.2,-.4,.1]]);np.testing.assert_allclose(v@rz_row(90),(rz_row(-90)@v.T).T,atol=1e-12)
 def test_normalization_fits_and_centers(self):
  v=np.array([[2,-3,4],[4,7,8],[3,0,5]],float);center,s,n=normalization(v);world=np.column_stack([v,np.ones(len(v))])@n.T;np.testing.assert_allclose(world[:,:3],(v-center)*s);self.assertAlmostEqual(np.ptp(world[:,:3],axis=0).max(),.99999);np.testing.assert_allclose(world[:,:3].min(0)+world[:,:3].max(0),0,atol=1e-12)
 def test_camera0_axes_and_rigid_inverse(self):
  c=camera_to_world(0);np.testing.assert_allclose(c[:3,0],[0,1,0]);np.testing.assert_allclose(c[:3,1],[0,0,1]);np.testing.assert_allclose(-c[:3,2],[-1,0,0]);np.testing.assert_allclose(np.linalg.inv(c)@c,np.eye(4));self.assertAlmostEqual(np.linalg.det(c[:3,:3]),1)
 def test_project_center_right_up(self):
  k,l=intrinsic();pts=np.array([[0,0,0],[0,.1,0],[0,0,.1]]);pixel,d=project_world(pts,np.linalg.inv(camera_to_world(0)),k);np.testing.assert_allclose(pixel[0],[256,256]);self.assertGreater(pixel[1,0],256);self.assertLess(pixel[2,1],256);np.testing.assert_allclose(d,2.2);self.assertAlmostEqual(l,16/math.tan(math.pi/12))
 def test_camera_all_views_projects_origin_center(self):
  k,_=intrinsic()
  for yaw in range(0,360,45):
   pixel,d=project_world(np.zeros((1,3)),np.linalg.inv(camera_to_world(yaw)),k);np.testing.assert_allclose(pixel,[[256,256]],atol=1e-10)
 def test_rgba_alpha_not_color_difference(self):
  rgba=np.zeros((8,8,4),np.uint8);rgba[2:6,1:7]=255;result=validate_alpha(rgba);self.assertEqual(result['bbox_strong_xyxy'],[1,2,6,5]);self.assertEqual(result['coverage_strong'],24/64)
 def test_empty_opaque_wrong_rgba_denied(self):
  for a in [np.zeros((4,4,4),np.uint8),np.full((4,4,4),255,np.uint8),np.zeros((4,4,3),np.uint8)]:
   with self.assertRaises(ValueError):validate_alpha(a)
 def test_review_passes_source_and_node_hash(self):
  with tempfile.TemporaryDirectory() as td:
   p,q,r=fixture(td);review,src,native=validate_review(q);self.assertEqual(src,p);self.assertEqual(native,r['reviewed_native_mesh_nodes'])
 def test_parent_asset_review_schema_and_matrix(self):
  with tempfile.TemporaryDirectory() as td:
   p,q,r=fixture(td);r.pop('reviewed_native_mesh_nodes');r.pop('source_coordinate_convention');r.pop('reviewer');r.pop('review_evidence');r.update(approved_for_render=True,approved_source_mesh_node_names=['Vehicle'],source_scene_topology={'nodes':[{'index':0,'name':'Vehicle','mesh':0}]},matrix_row_native_to_raw=native_to_raw_matrix(0).tolist(),review_scope='unit CPU fixture',manual_image_observation='fixture only')
   q.write_text(json.dumps(r));validate_review(q)
   r['source_scene_topology']['nodes'][0]['mesh']=1;q.write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'indices'):validate_review(q)
 def test_external_texture_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,q,r=fixture(td);doc=glb_json(p);doc['images']=[{'uri':'missing.png'}];raw=json.dumps(doc).encode();raw+=b' '*((-len(raw))%4);p.write_bytes(struct.pack('<III',0x46546c67,2,20+len(raw))+struct.pack('<II',len(raw),0x4e4f534a)+raw);r['source']['sha256']=sha(p);q.write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'external'):validate_review(q)
 def test_changed_source_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,q,r=fixture(td);p.write_bytes(p.read_bytes()+b'x')
   with self.assertRaisesRegex(ValueError,'hash'):validate_review(q)
 def test_unapproved_or_node_subset_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,q,r=fixture(td);r['review_passed']=False;q.write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'passed'):validate_review(q)
   r['review_passed']=True;r['reviewed_native_mesh_nodes']=[];q.write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'whitelist'):validate_review(q)
 def test_renderer_will_not_cleanup(self):
  with tempfile.TemporaryDirectory() as td:
   p,q,r=fixture(td);r['retain_all_reviewed_meshes']=False;q.write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'never deletes'):validate_review(q)
 def test_output_overwrite_denied(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(FileExistsError):fresh_output(td)
 def test_cuda6_7_single_only(self):
  for a in ['','0','4','6,7','GPU-abc']:
   with self.assertRaises(ValueError):strict_cuda_environment(a)
  self.assertEqual(strict_cuda_environment('6'),'6');self.assertEqual(strict_cuda_environment('7'),'7')
 def test_behind_camera_denied(self):
  k,_=intrinsic()
  with self.assertRaises(ValueError):project_world([[3,0,0]],np.linalg.inv(camera_to_world(0)),k)

if __name__=='__main__':unittest.main(verbosity=2)
