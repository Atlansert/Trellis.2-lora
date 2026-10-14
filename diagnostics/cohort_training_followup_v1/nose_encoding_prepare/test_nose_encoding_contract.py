import unittest,tempfile,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nose_encoding_contract as c

class Gates(unittest.TestCase):
 def test_proper_180_and_up(self):
  R=c.matrix_check(np.diag([-1,-1,1]),True);np.testing.assert_array_equal([0,0,1]@R,[0,0,1]);self.assertEqual(np.linalg.det(R),1)
 def test_identity_only_for_reuse(self):
  c.matrix_check(np.eye(3),False)
  with self.assertRaises(ValueError):c.matrix_check(np.diag([-1,-1,1]),False)
 def test_reflection_denied(self):
  with self.assertRaises(ValueError):c.matrix_check(np.diag([1,-1,1]),True)
 def test_changed_condition_sha_refused(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'condition';p.write_bytes(b'old');r={'path':str(p),'sha256':c.sha(p)};p.write_bytes(b'new')
   with self.assertRaises(ValueError):c.artifact(r)
 def test_fresh_wrong_output_refused(self):
  with tempfile.TemporaryDirectory() as td:
   with patch.object(c,'validate_batch',return_value=({'candidate_manifest_sha256':'x'},[])):
    with self.assertRaises(FileExistsError):c.fresh_or_resume(td,'dummy')
 def test_identity_stage_output_hash_checked(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);x=p/'ss_mean.npz';x.write_bytes(b'old');c.atomic(p/'ss_COMPLETE.json',{'input_geometry_sha256':'g','files':{'ss_mean.npz':c.sha(x)},'result':{}});self.assertTrue(c.stage_done(p,'ss','g'));x.write_bytes(b'new')
   with self.assertRaises(ValueError):c.stage_done(p,'ss','g')
 def test_partial_stage_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_bytes(b'x')
   with self.assertRaises(FileExistsError):c.assert_new([p])
 def test_gpu_stage_requires_device(self):
  with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
   with self.assertRaises(ValueError):c.device_guard('encode')
 def test_execution_guard_precedes_import(self):
  s=Path(__file__).with_name('encode_nose_targets.py').read_text();self.assertLess(s.index('if not a.execute:'),s.index('import torch'))
if __name__=='__main__':unittest.main(verbosity=2)
