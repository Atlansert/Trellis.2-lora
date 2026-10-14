"""SS 规范契约（build_ss_canonical_contract.py）的 CPU 单元测试。

- 覆盖点：姿态判据（长度轴必须落在 index1，深度轴不行）、非法体素坐标抛错、
  目标 latent 的 dtype/形状/有限性校验、帧路径的目录穿越防护（'..' 必须拒绝）、
  输出目录已存在时拒绝覆盖（FileExistsError）、非有限 max_angle 拒绝。
- 运行：python -m pytest tests/test_ss_canonical_contract.py（纯 CPU，无需 GPU/数据）。
- 依赖：numpy、PIL、data_toolkit.build_ss_canonical_contract。
"""
import unittest,tempfile,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
from data_toolkit.build_ss_canonical_contract import pose,valid_pose,read_mean,relative_frame,stage

class ContractTests(unittest.TestCase):
    # 辅助：构造 4x20x3 的长轴体素块；axis='wrong' 时把长度轴挪到轴 0（应被拒）。
    def coords(self,axis):
        c=np.array([[i,j,k] for i in range(4) for j in range(20) for k in range(3)],dtype=np.int16)
        return c[:,[1,2,0]] if axis=='wrong' else c
    # 契约：长度轴在 index1 时姿态合格（angle≤10 且主轴跨度在 index1）；把它换到轴 0 就该被拒。
    def test_pose_accepts_length1_not_depth(self):
        # Native voxel horizontal axis0 maps SS length1.
        c=np.array([[i,j,k] for i in range(50) for j in range(3) for k in range(4)])
        self.assertTrue(valid_pose(pose(c),10))
        self.assertFalse(valid_pose(pose(c[:,[1,0,2]]),10))
    # 契约：非 Nx3、非整数、超出 grid64 的坐标一律抛 ValueError。
    def test_invalid_coords(self):
        for c in [np.zeros((3,3)),np.ones((8,4)),np.ones((8,3))*64,np.ones((8,3))*.5]:
            with self.assertRaises(ValueError):pose(c)
    # 契约：目标必须是 fp32、形状 (8,16,16,16)、且全有限；float64/形状不符/含 NaN 都要拒绝。
    def test_target_dtype_shape_finite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.npz'
            for a in [np.zeros((8,16,16,16),np.float64),np.zeros((8,16,16,15),np.float32),np.full((8,16,16,16),np.nan,np.float32)]:
                np.savez(p,mean=a)
                with self.assertRaises(ValueError):read_mean(p)
    # 契约：带 '..' 的帧路径属于目录穿越，必须拒绝。
    def test_frame_traversal(self):
        with self.assertRaises(ValueError):relative_frame(Path('.'),{'file_path':'../image.png'})
    # 契约：输出目录已存在时必须 FileExistsError（防止覆盖已有暂存结果）。
    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):stage(td,td,10)
    # 契约：非有限的 max_angle 必须被拒绝（不允许 NaN 绕过阈值比较）。
    def test_nan_angle_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):stage(td,Path(td)/'out',float('nan'))

if __name__=='__main__':unittest.main(verbosity=2)
