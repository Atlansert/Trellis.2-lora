"""规范帧（canonical frame）坐标变换的 CPU 单元测试。

- 覆盖点：native glTF Y-up → 内部 (x,-z,y) 的映射、yaw 旋转是纯旋转（det=+1、竖直轴不变）、
  内部帧 → 正式 GLB (x,z,-y) 的往返一致、旧 SS 体素支撑集映射是网格双射（对合）而非特征变换、
  非法输入（越界/非整数/NaN/Inf）必须抛 ValueError。
- 运行：python -m pytest tests/test_canonical_frame.py 或 python tests/test_canonical_frame.py（纯 CPU，无需 GPU）。
- 依赖：numpy 与 data_toolkit.canonical_frame（不依赖任何数据或权重）。
"""
import unittest
import numpy as np
from data_toolkit.canonical_frame import native_gltf_to_internal,legacy_ss_support_to_internal,internal_to_formal_glb

class FrameTests(unittest.TestCase):
 # 契约：轴标记向量经规范帧变换后，up/depth/length 三个槽位必须落在预期位置，并可逆回原生。
 def test_marker_up_depth_length_match_actual_blender(self):
  v=np.diag([.125,.25,.5]);expected=np.array([[.125,0,0],[0,0,.25],[0,-.5,0]])
  np.testing.assert_allclose(native_gltf_to_internal(v),expected)
  np.testing.assert_allclose(internal_to_formal_glb(expected),v)
 # 契约：所有 yaw 下的变换都是纯旋转（R·Rᵀ=I、det=+1），且世界竖直轴始终映射到内部 index1。
 def test_all_view_transforms_proper_and_up_preserved(self):
  for yaw in range(0,360,15):
   r=native_gltf_to_internal(np.eye(3),yaw)
   np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12);self.assertAlmostEqual(np.linalg.det(r),1.)
   np.testing.assert_allclose(r[1],[0,0,1],atol=1e-12)
 # 契约：原生沿 X 的长轴在 yaw=90° 时应落在内部 -Y 方向。
 def test_native_long_x_yaw90_becomes_internal_y(self):
  np.testing.assert_allclose(native_gltf_to_internal(np.array([[1,0,0]]),90),[[0,-1,0]],atol=1e-12)
 # 契约：旧 SS 支撑集映射是网格上的双射（对合），再次调用必须还原原始坐标。
 def test_support_conversion_is_grid_bijection_not_feature_transform(self):
  c=np.array([[0,1,2],[63,62,61]],np.uint8);x=legacy_ss_support_to_internal(c)
  np.testing.assert_array_equal(x,[[61,1,63],[2,62,0]]);np.testing.assert_array_equal(legacy_ss_support_to_internal(x),c)
 # 契约：越界、非整数、NaN/Inf 的输入与非法 yaw 必须直接失败，不做隐式兜底。
 def test_invalid_support_and_vertices_fail(self):
  for c in [np.array([[64,0,0]]),np.array([[.5,0,0]]),np.array([[np.nan,0,0]])]:
   with self.assertRaises(ValueError):legacy_ss_support_to_internal(c)
  with self.assertRaises(ValueError):native_gltf_to_internal([[0,0,float('inf')]])
  with self.assertRaises(ValueError):native_gltf_to_internal([[0,0,0]],float('nan'))
if __name__=='__main__':unittest.main(verbosity=2)
