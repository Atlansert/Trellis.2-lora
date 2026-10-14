import json,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import batch_encoding_contract as bc
class Tests(unittest.TestCase):
 def fixture(self,td):
  root=Path(td);source=root/'src.glb';source.write_bytes(b'source');asset_review=root/'review.json';asset_review.write_text('{}');rows=[]
  for obj in sorted(bc.IDS):
   d=root/obj;d.mkdir();v=np.array([[-.1,-.49,-.1],[.1,.49,.1],[-.1,.2,.1]],np.float32);f=np.array([[0,1,2]],np.int32);np.savez(d/'canonical_geometry.npz',vertices=v,faces=f)
   views=[]
   for i in range(8):p=d/f'{i:03d}.png';p.write_bytes(b'png'+bytes([i]));views.append({'index':i,'file':p.name,'sha256':bc.sha(p)})
   m={'status':'success','training_approved':False,'all_reviewed_source_meshes_retained':True,'options':{'resolution':512},'source_path':str(source),'source_sha256':bc.sha(source),'review_path':str(asset_review),'review_sha256':bc.sha(asset_review),'views':views,'geometry':{'path':'canonical_geometry.npz','sha256':bc.sha(d/'canonical_geometry.npz')}}
   (d/'manifest.json').write_text(json.dumps(m));(d/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':bc.sha(d/'manifest.json'),'views':8}));rows.append({'object_id':obj,'render_dir':str(d),'render_manifest_sha256':bc.sha(d/'manifest.json')})
  p=root/'batch_render_validation.json';p.write_text(json.dumps({'schema':'trellis.batch-render-validation.v1','passed':True,'approved_for_encoding':True,'assets':rows}));return p,rows,source
 def test_valid_six_assets_without_models(self):
  with tempfile.TemporaryDirectory() as td:
   p,_,_=self.fixture(td);_,rows=bc.validate_batch(p);self.assertEqual(len(rows),6)
 def test_explicit_approval_required(self):
  with tempfile.TemporaryDirectory() as td:
   p,_,_=self.fixture(td);m=json.loads(p.read_text());m['approved_for_encoding']=False;p.write_text(json.dumps(m))
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_manifest_complete_hash_change(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows,_=self.fixture(td);q=Path(rows[0]['render_dir'])/'manifest.json';q.write_text(q.read_text()+' ')
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_source_changed(self):
  with tempfile.TemporaryDirectory() as td:
   p,_,src=self.fixture(td);src.write_bytes(b'changed')
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_image_changed(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows,_=self.fixture(td);(Path(rows[0]['render_dir'])/'000.png').write_bytes(b'changed')
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_geometry_changed(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows,_=self.fixture(td);q=Path(rows[0]['render_dir'])/'canonical_geometry.npz';q.write_bytes(q.read_bytes()+b'x')
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_missing_view_or_scope(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows,_=self.fixture(td);m=json.loads(p.read_text());m['assets'].pop();p.write_text(json.dumps(m))
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_missing_view_rehashed_contract_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows,_=self.fixture(td);d=Path(rows[0]['render_dir']);m=json.loads((d/'manifest.json').read_text());m['views'].pop();(d/'manifest.json').write_text(json.dumps(m));c=json.loads((d/'COMPLETE.json').read_text());c['manifest_sha256']=bc.sha(d/'manifest.json');(d/'COMPLETE.json').write_text(json.dumps(c));batch=json.loads(p.read_text());batch['assets'][0]['render_manifest_sha256']=bc.sha(d/'manifest.json');p.write_text(json.dumps(batch))
   with self.assertRaises(ValueError):bc.validate_batch(p)
 def test_parent_objects_render_directory_schema(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows,_=self.fixture(td);batch=json.loads(p.read_text());batch['objects']=[{'object_id':r['object_id'],'render_directory':r['render_dir'],'render_manifest_sha256':r['render_manifest_sha256']} for r in batch.pop('assets')];p.write_text(json.dumps(batch));self.assertEqual(len(bc.validate_batch(p)[1]),6)
 def test_resume_model_dependency_hash_changed(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);p=d/'shape.npz';p.write_bytes(b'x');prefix=d/'encoder';Path(str(prefix)+'.json').write_text('{}');Path(str(prefix)+'.safetensors').write_bytes(b'weights');result={'checkpoint':{'prefix':str(prefix),'config_sha256':bc.sha(str(prefix)+'.json'),'weight_sha256':bc.sha(str(prefix)+'.safetensors')}};bc.stage_commit(d,'encode','a',[p],result);self.assertTrue(bc.stage_done(d,'encode','a'));Path(str(prefix)+'.safetensors').write_bytes(b'changed')
   with self.assertRaises(ValueError):bc.stage_done(d,'encode','a')
 def test_nonfinite_and_wrong_axis_geometry(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'g.npz';f=np.array([[0,1,2]])
   for v in [np.array([[0,0,0],[float('nan'),.4,0],[.1,-.4,0]]),np.array([[-.49,0,0],[.49,.1,0],[0,0,.1]])]:
    np.savez(p,vertices=v,faces=f)
    with self.assertRaises(ValueError):bc.load_geometry(p)
 def test_unsafe_and_output_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(ValueError):bc.bound_file(td,'../source')
   p,_,_=self.fixture(td);out=Path(td)/'existing';out.mkdir()
   with self.assertRaises(FileExistsError):bc.fresh_or_resume(out,p)
 def test_committed_stage_resume_checks_hash(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);p=d/'shape.npz';p.write_bytes(b'x');bc.stage_commit(d,'encode','a',[p],{});self.assertTrue(bc.stage_done(d,'encode','a'));p.write_bytes(b'y')
   with self.assertRaises(ValueError):bc.stage_done(d,'encode','a')
 def test_partial_output_refused(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_bytes(b'x')
   with self.assertRaises(FileExistsError):bc.assert_new([p])
 def test_gpu_stage_restrictions(self):
  for stage,value in [('encode','0'),('decode','5'),('ss','6')]:
   with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=value):
    with self.assertRaises(ValueError):bc.device_guard(stage)
if __name__=='__main__':unittest.main(verbosity=2)
