import unittest,tempfile,json,hashlib
from pathlib import Path
import test_ss_canonical_integration as fixtures
from data_toolkit.build_ss_canonical_contract import stage
from trellis2.datasets.canonical_contract import load_canonical_contract,file_sha256
from trellis2.datasets.sparse_structure_latent_views import ImageConditionedSparseStructureLatentViews as Dataset

class LoadingTests(unittest.TestCase):
 def fixture(self,td):
  r=fixtures.Integration().fixture(td);out=Path(td)/'contract';stage(r,out,10);return r,out
 def test_staged_denied_by_default(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td)
   with self.assertRaisesRegex(ValueError,'not approved'):load_canonical_contract(c/'manifest.json',c/'canonical')
 def test_explicit_diagnostic_open(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);rr,o=load_canonical_contract(c/'manifest.json',c/'canonical',True);self.assertEqual(set(o),{'native'});self.assertEqual(rr,r.resolve())
 def test_manifest_tamper_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);p=c/'manifest.json';p.write_text(p.read_text()+' ')
   with self.assertRaisesRegex(ValueError,'checksum'):load_canonical_contract(p,c/'canonical',True)
 def test_source_image_tamper_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);p=r/'renders_cond_t/native/000.png';p.write_bytes(p.read_bytes()+b'x')
   with self.assertRaisesRegex(ValueError,'source changed'):load_canonical_contract(c/'manifest.json',c/'canonical',True)
 def test_missing_complete_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);(c/'COMPLETE.json').unlink()
   with self.assertRaises(FileNotFoundError):load_canonical_contract(c/'manifest.json',c/'canonical',True)
 def test_approval_boolean_without_evidence_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);(c/'TRAINING_APPROVAL.json').write_text(json.dumps({'approved':True,'manifest_sha256':file_sha256(c/'manifest.json')}))
   with self.assertRaisesRegex(ValueError,'approval'):load_canonical_contract(c/'manifest.json',c/'canonical')
 def test_root_mismatch_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td)
   with self.assertRaisesRegex(ValueError,'root'):load_canonical_contract(c/'manifest.json',r,True)
 def test_dataset_contract_views_and_staged_guard(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);roots={'test':{'base':str(r),'render_cond':str(r/'renders_cond_t'),'ss_view_latent':str(r/'ss_latents_views_canon')}};kwargs=dict(min_aesthetic_score=4.5,num_views=24,latent_frame='canonical',canonical_latent_root=str(c/'canonical'),canonical_manifest=str(c/'manifest.json'),max_view_offaxis=10)
   with self.assertRaisesRegex(ValueError,'not approved'):Dataset(roots,**kwargs)
   d=Dataset(roots,allow_staged_contract=True,**kwargs);self.assertEqual(len(d),1);self.assertEqual(d.instances[0][3],[0,12])
   sample=d[0];self.assertEqual(tuple(sample['x_0'].shape),(8,16,16,16));self.assertEqual(tuple(sample['cond'].shape),(3,512,512))
   target=c/'canonical/native.npz';original=target.read_bytes();target.write_bytes(original+b'x')
   with self.assertRaisesRegex(ValueError,'after initialization'):d[0]
   target.write_bytes(original)
   kwargs['min_aesthetic_score']=6
   with self.assertRaisesRegex(ValueError,'object set'):Dataset(roots,allow_staged_contract=True,**kwargs)
   kwargs['min_aesthetic_score']=4.5
   kwargs['num_views']=1
   with self.assertRaisesRegex(ValueError,'filters change'):Dataset(roots,allow_staged_contract=True,**kwargs)
if __name__=='__main__':unittest.main(verbosity=2)
