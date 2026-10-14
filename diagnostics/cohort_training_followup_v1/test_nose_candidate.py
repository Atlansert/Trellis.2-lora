import unittest,json,hashlib
from pathlib import Path
import numpy as np
B=Path('/data/lzx/TRELLIS.2/diagnostics/cohort_training_followup_v1/nose_canonical_candidate')
class NoseCandidateTests(unittest.TestCase):
 def setUp(self):self.m=json.loads((B/'manifest.json').read_text())
 def test_scope_and_no_approval(self):
  self.assertEqual(len(self.m['records']),6);self.assertFalse(self.m['training_approved']);self.assertTrue(self.m['geometry_only']);self.assertTrue(self.m['encoded_latent_permutation_forbidden'])
 def test_only_source_positive_nose_rotates(self):
  ids={r['object_id'] for r in self.m['records'] if r['geometry_candidate']};self.assertEqual(ids,{'1208e4cc0c30448a','1f7cc1ae1cec73f5'})
 def test_proper_up_involution(self):
  for r in self.m['records']:
   R=np.asarray(r['matrix_row_oldraw_to_canonical']);self.assertAlmostEqual(np.linalg.det(R),1);np.testing.assert_allclose(R@R,np.eye(3));np.testing.assert_allclose(np.array([0,0,1])@R,[0,0,1])
 def test_actual_faces_extents_and_normals(self):
  for r in self.m['records']:
   if not r['geometry_candidate']:continue
   with np.load(r['source_geometry']['path']) as z:v=z['vertices'];f=z['faces']
   with np.load(r['geometry_candidate']['path']) as z:n=z['vertices'];g=z['faces']
   self.assertTrue(np.array_equal(f,g));np.testing.assert_allclose(np.ptp(v,axis=0),np.ptp(n,axis=0));np.testing.assert_array_equal(v[:,2],n[:,2]);R=np.asarray(r['matrix_row_oldraw_to_canonical']);i=f[::1000];np.testing.assert_allclose(np.cross(n[i[:,1]]-n[i[:,0]],n[i[:,2]]-n[i[:,0]]),np.cross(v[i[:,1]]-v[i[:,0]],v[i[:,2]]-v[i[:,0]])@R,atol=1e-9)
 def test_source_and_conditions_immutable(self):
  for r in self.m['records']:
   for rec in [r['source_geometry'],r['source_condition']]:self.assertEqual(hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest(),rec['sha256'])
 def test_condition_camera_transform_explicit(self):
  for r in self.m['records']:
   R=np.asarray(r['matrix_row_oldraw_to_canonical']);T=np.eye(4);T[:3,:3]=R.T;np.testing.assert_allclose(T@np.asarray(r['camera0_to_world_oldraw']),r['same_image_camera_to_world_newcanonical'])
 def test_no_encoded_ready_targets(self):
  for r in self.m['records']:self.assertIsNone(r['new_shape_latent']);self.assertIsNone(r['new_ss_mean']);self.assertFalse(r['training_approved'])
 def test_completion_hash(self):
  c=json.loads((B/'COMPLETE.json').read_text());self.assertEqual(c['manifest_sha256'],hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest())
if __name__=='__main__':unittest.main(verbosity=2)
