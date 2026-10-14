import unittest,tempfile,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
from build_canonical_contract import pose,valid_pose,read_mean,relative_frame,stage

class ContractTests(unittest.TestCase):
    def coords(self,axis):
        c=np.array([[i,j,k] for i in range(4) for j in range(20) for k in range(3)],dtype=np.int16)
        return c[:,[1,2,0]] if axis=='wrong' else c
    def test_pose_accepts_length1_not_depth(self):
        # Native voxel horizontal axis0 maps SS length1.
        c=np.array([[i,j,k] for i in range(50) for j in range(3) for k in range(4)])
        self.assertTrue(valid_pose(pose(c),10))
        self.assertFalse(valid_pose(pose(c[:,[1,0,2]]),10))
    def test_invalid_coords(self):
        for c in [np.zeros((3,3)),np.ones((8,4)),np.ones((8,3))*64,np.ones((8,3))*.5]:
            with self.assertRaises(ValueError):pose(c)
    def test_target_dtype_shape_finite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.npz'
            for a in [np.zeros((8,16,16,16),np.float64),np.zeros((8,16,16,15),np.float32),np.full((8,16,16,16),np.nan,np.float32)]:
                np.savez(p,mean=a)
                with self.assertRaises(ValueError):read_mean(p)
    def test_frame_traversal(self):
        with self.assertRaises(ValueError):relative_frame(Path('.'),{'file_path':'../image.png'})
    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):stage(td,td,10)
    def test_nan_angle_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):stage(td,Path(td)/'out',float('nan'))

if __name__=='__main__':unittest.main(verbosity=2)
