import unittest
import numpy as np
from data_toolkit.canonical_frame import native_gltf_to_internal,legacy_ss_support_to_internal,internal_to_formal_glb

class FrameTests(unittest.TestCase):
 def test_marker_up_depth_length_match_actual_blender(self):
  v=np.diag([.125,.25,.5]);expected=np.array([[.125,0,0],[0,0,.25],[0,-.5,0]])
  np.testing.assert_allclose(native_gltf_to_internal(v),expected)
  np.testing.assert_allclose(internal_to_formal_glb(expected),v)
 def test_all_view_transforms_proper_and_up_preserved(self):
  for yaw in range(0,360,15):
   r=native_gltf_to_internal(np.eye(3),yaw)
   np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12);self.assertAlmostEqual(np.linalg.det(r),1.)
   np.testing.assert_allclose(r[1],[0,0,1],atol=1e-12)
 def test_native_long_x_yaw90_becomes_internal_y(self):
  np.testing.assert_allclose(native_gltf_to_internal(np.array([[1,0,0]]),90),[[0,-1,0]],atol=1e-12)
 def test_support_conversion_is_grid_bijection_not_feature_transform(self):
  c=np.array([[0,1,2],[63,62,61]],np.uint8);x=legacy_ss_support_to_internal(c)
  np.testing.assert_array_equal(x,[[61,1,63],[2,62,0]]);np.testing.assert_array_equal(legacy_ss_support_to_internal(x),c)
 def test_invalid_support_and_vertices_fail(self):
  for c in [np.array([[64,0,0]]),np.array([[.5,0,0]]),np.array([[np.nan,0,0]])]:
   with self.assertRaises(ValueError):legacy_ss_support_to_internal(c)
  with self.assertRaises(ValueError):native_gltf_to_internal([[0,0,float('inf')]])
  with self.assertRaises(ValueError):native_gltf_to_internal([[0,0,0]],float('nan'))
if __name__=='__main__':unittest.main(verbosity=2)
