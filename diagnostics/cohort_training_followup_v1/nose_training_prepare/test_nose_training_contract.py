import unittest,tempfile,json
from pathlib import Path
import numpy as np
from nose_training_contract import artifact,sha,finite_mean,rotation,fresh,approval_check,TRAIN,HOLDOUT,Dataset
class Gates(unittest.TestCase):
 def fixture(self,td):
  b=Path(td);paths=[b/n for n in ['old','new','candidate','baseline']]
  for p in paths:p.write_text('{}')
  gate=b/'gate.json';gate.write_text(json.dumps({'schema':'trellis.nose-target-independent-validation.v1','approved_for_scoped_target_trial':True,'production_approved':False,'conditions_changed':False,'passed':True,'encoded_manifest_sha256':sha(paths[1]),'objects':TRAIN+HOLDOUT}))
  a={'schema':'trellis.nose-canonical-target-trial-approval.v1','approved':True,'diagnostic_only':True,'production_approved':False,'max_steps':100,'train_objects':TRAIN.copy(),'internal_holdout_objects':HOLDOUT.copy(),'condition_view':0,'base_sha256':'base','target_validation':{'path':str(gate),'sha256':sha(gate)}}
  for k,p in zip(['old_manifest_sha256','new_target_manifest_sha256','candidate_manifest_sha256','old_baseline100_run_contract_sha256'],paths):a[k]=sha(p)
  return a,paths,gate
 def test_rotation_proper_identity_and180(self):rotation(np.eye(3),False);rotation(np.diag([-1,-1,1]),True)
 def test_reflection_rejected(self):
  with self.assertRaises(ValueError):rotation(np.diag([1,-1,1]),True)
 def test_mean_nan_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'mean.npz';np.savez(p,mean=np.full((8,16,16,16),np.nan,np.float32))
   with self.assertRaises(ValueError):finite_mean(p)
 def test_mean_dtype_and_shape_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'mean.npz'
   for a in [np.zeros((8,16,16,16),np.float64),np.zeros((8,15,16,16),np.float32)]:
    np.savez(p,mean=a)
    with self.assertRaises(ValueError):finite_mean(p)
 def test_artifact_changed_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_text('a');r={'path':str(p),'sha256':sha(p)};p.write_text('b')
   with self.assertRaises(ValueError):artifact(r)
 def test_unsafe_path_rejected(self):
  with self.assertRaises(ValueError):artifact({'path':'../x','sha256':'x'})
 def test_output_overwrite_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(FileExistsError):fresh(td)
 def test_valid_scoped_new_approval(self):
  with tempfile.TemporaryDirectory() as td:
   a,ps,g=self.fixture(td);approval_check(a,*ps,'base')
 def test_200_and_old_approval_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   a,ps,g=self.fixture(td);a['max_steps']=200
   with self.assertRaises(ValueError):approval_check(a,*ps,'base')
   a['max_steps']=100;a['schema']='trellis.six-render-ss-approval.v1'
   with self.assertRaises(ValueError):approval_check(a,*ps,'base')
 def test_holdout_leak_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   a,ps,g=self.fixture(td);a['train_objects']=TRAIN+[HOLDOUT[0]]
   with self.assertRaises(ValueError):approval_check(a,*ps,'base')
 def test_validation_scope_and_checksum_fail(self):
  with tempfile.TemporaryDirectory() as td:
   a,ps,g=self.fixture(td);g.write_text(json.dumps({'schema':'trellis.nose-target-independent-validation.v1','approved_for_scoped_target_trial':True,'production_approved':False,'conditions_changed':False,'passed':True,'encoded_manifest_sha256':sha(ps[1]),'objects':TRAIN}));a['target_validation']['sha256']=sha(g)
   with self.assertRaises(ValueError):approval_check(a,*ps,'base')
   a,ps,g=self.fixture(td);ps[1].write_text('changed')
   with self.assertRaises(ValueError):approval_check(a,*ps,'base')
 def test_dataset_fourdraw_slots_not_holdout(self):self.assertEqual(TRAIN.__len__(),4);self.assertFalse(set(TRAIN)&set(HOLDOUT));self.assertTrue(Dataset.skip_snapshot)
if __name__=='__main__':unittest.main(verbosity=2)
