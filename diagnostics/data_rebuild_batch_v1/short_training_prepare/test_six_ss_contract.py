import unittest,tempfile,json,sys,ast
from pathlib import Path
import numpy as np
from PIL import Image
import six_ss_contract as c

class ContractTests(unittest.TestCase):
 def fixture(self,td):
  b=Path(td).resolve();objects=[]
  dummy=b/'encoded.json';dummy.write_text('{}')
  render=b/'render.json';render.write_text(json.dumps({'status':'success','raw_axes':{'length':'y','up':'z'}}))
  mean=b/'mean.npz';np.savez(mean,mean=np.zeros((8,16,16,16),np.float32))
  geom=b/'geometry.npz';np.savez(geom,vertices=np.zeros((3,3)))
  review=b/'review.json';review.write_text('{}')
  rgba=b/'rgba.png';a=np.zeros((16,16,4),np.uint8);a[5:11,2:14]=[200,100,50,255];Image.fromarray(a).save(rgba)
  rgb=b/'rgb.png';c.production_condition(Image.fromarray(a)).save(rgb)
  for obj in c.TRAIN+c.HOLDOUT:
   objects.append({'object_id':obj,'split':'train' if obj in c.TRAIN else 'internal_holdout','view_index':0,'render_rgba':c.record(rgba),'render_manifest':c.record(render),'canonical_geometry':c.record(geom),'target':c.record(mean),'condition':c.record(rgb),'source_review':c.record(review)})
  m={'schema':c.SCHEMA,'production_training_approved':False,'target_frame':'proper_raw_depth_length_up','train_objects':list(c.TRAIN),'internal_holdout_objects':list(c.HOLDOUT),'objects':objects,'dependencies':{'encoded_manifest':c.record(dummy)}}
  p=b/'manifest.json';self.save(p,m);return p,m
 def save(self,p,m):
  p.write_text(json.dumps(m));p.with_name('COMPLETE.json').write_text(json.dumps({'manifest_sha256':c.sha(p)}))
 def approval(self,p):
  proof=p.with_name('proof.json');proof.write_text(json.dumps({'passed':True,'scope':list(c.TRAIN+c.HOLDOUT)}))
  g=p.with_name('gate.json');g.write_text(json.dumps({'schema':'trellis.six-target-scoped-data-gate.v1','data_validation_passed':True,'approved_for_short_training':True,'objects':list(c.TRAIN+c.HOLDOUT),'approved_condition_views':[0],'production_approved':False,'validation_path':str(proof),'validation_sha256':c.sha(proof)}))
  visual=p.with_name('visual.json');visual.write_text(json.dumps({'schema':'trellis.render-batch-visual-review.v1','passed':True,'objects':list(c.TRAIN+c.HOLDOUT),'reviewed_condition_views':[0]}))
  split=p.with_name('split.json');split.write_text(json.dumps({'schema':'trellis.small-cohort-split.v1','train':list(c.TRAIN),'heldout':list(c.HOLDOUT),'condition_views':[0],'first_phase_steps':50,'resume_to100_automatic':False}))
  a={'schema':'trellis.six-render-ss-approval.v1','manifest_sha256':c.sha(p),'approved':True,'diagnostic_only':True,'train_objects':list(c.TRAIN),'internal_holdout_objects':list(c.HOLDOUT),'first_phase_steps':50,'total_steps_limit':100,'target_validation':c.record(g),'render_visual_review':c.record(visual),'split_evidence':c.record(split),'encoded_manifest_sha256':json.loads(p.read_text())['dependencies']['encoded_manifest']['sha256']};q=p.with_name('approval.json');q.write_text(json.dumps(a));return q,a
 def test_cpu_validation_and_no_implicit_approval(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);self.assertEqual(len(c.validate(p,diagnostic=True)[1]),6)
   with self.assertRaisesRegex(ValueError,'approval'):c.validate(p,diagnostic=False)
 def test_exact_train_holdout_dataset_scope(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);q,a=self.approval(p);d=c.ManifestDataset(p,q);self.assertEqual(len(d),4);self.assertEqual(d.objects,list(c.TRAIN));self.assertFalse(set(d.objects)&set(c.HOLDOUT));self.assertEqual(tuple(d[0]['cond'].shape),(3,512,512));self.assertEqual(tuple(d[0]['x_0'].shape),(8,16,16,16));self.assertEqual(tuple(d.visualize_sample({'x_0':d[0]['x_0'][None]}).shape),(1,3,16,16))
 def test_holdout_leak_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['train_objects'].append(c.HOLDOUT[0]);self.save(p,m)
   with self.assertRaisesRegex(ValueError,'split'):c.validate(p)
 def test_source_checksum_change_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);Path(m['objects'][0]['render_rgba']['path']).write_bytes(b'x')
   with self.assertRaisesRegex(ValueError,'checksum'):c.validate(p)
 def test_manifest_hash_tamper_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);p.write_text(p.read_text()+' ')
   with self.assertRaisesRegex(ValueError,'identity'):c.validate(p)
 def test_nan_target_even_rehashed_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);target=Path(m['objects'][0]['target']['path']);np.savez(target,mean=np.full((8,16,16,16),np.nan,np.float32))
   for obj in m['objects']:obj['target']=c.record(target)
   self.save(p,m)
   with self.assertRaisesRegex(ValueError,'finite'):c.validate(p)
 def test_path_traversal_denied(self):
  with self.assertRaisesRegex(ValueError,'path'):c.verify({'path':'/tmp/../x','sha256':'0'*64,'size_bytes':1})
 def test_wrong_frame_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['target_frame']='legacy';self.save(p,m)
   with self.assertRaisesRegex(ValueError,'frame'):c.validate(p)
 def test_extra_image_views_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);m['objects'][0]['view_index']=4;self.save(p,m)
   with self.assertRaisesRegex(ValueError,'views'):c.validate(p)
 def test_phase100_not_implemented(self):
  self.assertEqual(c.phase_steps(50,100),50)
  for x in [0,51,100,True]:
   with self.assertRaises(ValueError):c.phase_steps(x,100)
 def test_output_overwrite_denied(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(FileExistsError):c.fresh_output(td)
 def test_parent_gate_failed_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);q,a=self.approval(p);g=Path(a['target_validation']['path']);d=json.loads(g.read_text());d['data_validation_passed']=False;g.write_text(json.dumps(d));a['target_validation']=c.record(g);q.write_text(json.dumps(a))
   with self.assertRaisesRegex(ValueError,'validation'):c.validate(p,q,False)
 def test_gate_wrong_view_scope_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);q,a=self.approval(p);g=Path(a['target_validation']['path']);d=json.loads(g.read_text());d['approved_condition_views']=[0,4];g.write_text(json.dumps(d));a['target_validation']=c.record(g);q.write_text(json.dumps(a))
   with self.assertRaisesRegex(ValueError,'scope'):c.validate(p,q,False)
 def test_visual_failed_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);q,a=self.approval(p);g=Path(a['render_visual_review']['path']);d=json.loads(g.read_text());d['passed']=False;g.write_text(json.dumps(d));a['render_visual_review']=c.record(g);q.write_text(json.dumps(a))
   with self.assertRaisesRegex(ValueError,'visual'):c.validate(p,q,False)
 def test_split_evidence_leak_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);q,a=self.approval(p);g=Path(a['split_evidence']['path']);d=json.loads(g.read_text());d['train'].append(c.HOLDOUT[0]);g.write_text(json.dumps(d));a['split_evidence']=c.record(g);q.write_text(json.dumps(a))
   with self.assertRaisesRegex(ValueError,'split'):c.validate(p,q,False)
 def test_validation_proof_tamper_denied(self):
  with tempfile.TemporaryDirectory() as td:
   p,m=self.fixture(td);q,a=self.approval(p);g=json.loads(Path(a['target_validation']['path']).read_text());Path(g['validation_path']).write_text('tampered')
   with self.assertRaisesRegex(ValueError,'proof'):c.validate(p,q,False)
 def test_condition_opaque_empty_fail(self):
  for alpha in [0,255]:
   im=Image.fromarray(np.full((16,16,4),alpha,np.uint8))
   with self.assertRaises(ValueError):c.production_condition(im)
 def test_rgba_preprocess_matches_actual_pipeline_math(self):
  root=Path('/data/lzx/TRELLIS.2');tree=ast.parse((root/'trellis2/pipelines/trellis2_image_to_3d.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Trellis2ImageTo3DPipeline');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image');env={'Image':Image,'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'live-pipeline-preprocess','exec'),env)
  rgba=np.zeros((64,64,4),np.uint8);rgba[22:41,4:58,:]=[201,105,35,255];rgba[20:22,6:54,:]=[201,105,35,128];im=Image.fromarray(rgba)
  expected=env['preprocess_image'](object(),im).resize((512,512),Image.Resampling.LANCZOS);actual=c.production_condition(im);self.assertTrue(np.array_equal(np.asarray(actual),np.asarray(expected)))
if __name__=='__main__':unittest.main(verbosity=2)
