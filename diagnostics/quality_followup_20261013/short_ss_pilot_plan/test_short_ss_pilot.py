import unittest,tempfile,json,copy
from pathlib import Path
import numpy as np
from PIL import Image
from short_ss_pilot_contract import sha,validate,ManifestDataset,fresh_output,checked_phase_steps

class PilotTests(unittest.TestCase):
 def fixture(self,td):
  r=Path(td);im=r/'c.png';Image.new('RGB',(512,512)).save(im);a=r/'a.npz';np.savez(a,mean=np.zeros((8,16,16,16),np.float32));rec=lambda p:{'path':str(p),'sha256':sha(p)}
  objs=[{'object':o,'condition':rec(im),'targets':{'legacy':rec(a),'proper':rec(a)},'target_frames':{'legacy':'legacy_ss_down_length_depth','proper':'proper_raw_depth_length_up'}} for o in ['0086bbab37559442','0fab4af3f9a73bdd']]
  m={'schema':'trellis.short-ss-pilot.v1','production_approved':False,'objects':objs};p=r/'manifest.json';p.write_text(json.dumps(m));(r/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha(p)}));return p,m
 def commit(self,p,m):
  p.write_text(json.dumps(m));p.with_name('COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha(p)}))
 def test_valid_cpu_dataset_same_condition(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);a=ManifestDataset(p,'legacy',diagnostic_only=True);b=ManifestDataset(p,'proper',diagnostic_only=True);self.assertEqual(len(a),2);self.assertTrue((a[0]['cond']==b[0]['cond']).all());self.assertEqual(tuple(a[0]['cond'].shape),(3,512,512))
 def test_parent_approval_missing(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td)
   with self.assertRaisesRegex(ValueError,'approval'):validate(p,'proper')
 def test_nan_mean_rejected_even_rehashed(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);a=Path(m['objects'][0]['targets']['proper']['path']);np.savez(a,mean=np.full((8,16,16,16),np.nan,np.float32))
   for o in m['objects']:
    for t in o['targets'].values():t['sha256']=sha(a)
   self.commit(p,m)
   with self.assertRaisesRegex(ValueError,'finite'):validate(p,'proper',diagnostic_only=True)
 def test_hash_change_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);Path(m['objects'][0]['condition']['path']).write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'checksum'):validate(p,'proper',diagnostic_only=True)
 def test_path_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['objects'][0]['condition']['path']=str(Path(td)/'../x');self.commit(p,m)
   with self.assertRaisesRegex(ValueError,'nontraversing'):validate(p,'proper',diagnostic_only=True)
 def test_target_frame_wrong_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['objects'][0]['target_frames']['proper']='legacy';self.commit(p,m)
   with self.assertRaisesRegex(ValueError,'frame'):validate(p,'proper',diagnostic_only=True)
 def test_approval_scope_false_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);a=Path(td)/'approval.json';a.write_text(json.dumps({'schema':'trellis.short-ss-pilot-approval.v1','manifest_sha256':sha(p),'approved':True,'diagnostic_only':True,'objects':['unknown']}))
   with self.assertRaisesRegex(ValueError,'scope'):validate(p,'proper',a)
 def test_quarantined_object_cannot_be_approved(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['approval_scope_limit']=['0086bbab37559442'];m['objects'][1]['execution_eligible']=False;self.commit(p,m);a=Path(td)/'approval.json';a.write_text(json.dumps({'schema':'trellis.short-ss-pilot-approval.v1','manifest_sha256':sha(p),'approved':True,'diagnostic_only':True,'objects':['0fab4af3f9a73bdd']}))
   with self.assertRaisesRegex(ValueError,'quarantined'):validate(p,'proper',a)
 def test_single_asset_approval_has_two_disclosed_draw_slots(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['approval_scope_limit']=['0086bbab37559442'];self.commit(p,m);e=Path(td)/'review.json';e.write_text(json.dumps({'passed':True,'objects':['0086bbab37559442']}));a=Path(td)/'approval.json';a.write_text(json.dumps({'schema':'trellis.short-ss-pilot-approval.v1','manifest_sha256':sha(p),'approved':True,'diagnostic_only':True,'objects':['0086bbab37559442'],'max_steps':100,'geometry_review_passed':True,'semantic_review_passed':True,'geometry_evidence':{'path':str(e),'sha256':sha(e)},'semantic_evidence':{'path':str(e),'sha256':sha(e)}}));d=ManifestDataset(p,'proper',a);self.assertEqual(len(d.objects),1);self.assertEqual(len(d),2);self.assertTrue((d[0]['x_0']==d[1]['x_0']).all())
 def test_phase_budget_and_invalid_requested(self):
  self.assertEqual(checked_phase_steps(50,100),50)
  for requested,approved in [(100,50),(200,100),(True,100),(50,0)]:
   with self.assertRaises(ValueError):checked_phase_steps(requested,approved)
 def test_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(FileExistsError):fresh_output(td)
if __name__=='__main__':unittest.main(verbosity=2)
