import ast,contextlib,io,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.utils.inference_contract import *

class Tensor:
 def __init__(self,shape):self.shape=shape
class FakeModel:
 def __init__(self,rope=True):self.state={'weight':Tensor((2,3))};self.calls=0;self.training=True;self.layers={};self.state.update({'rope_phases':Tensor((1,))} if rope else {})
 def state_dict(self):return self.state
 def load_state_dict(self,state,strict=False):self.calls+=1;return list(set(self.state)-set(state)),list(set(state)-set(self.state))
 def eval(self):self.training=False
 def get_submodule(self,name):return self.layers[name]
class ContractTests(unittest.TestCase):
 def test_rope_missing_allowed_and_eval(self):
  m=FakeModel();checked_load(m,{'weight':Tensor((2,3))});self.assertFalse(m.training)
 def test_bad_missing_rejected_before_mutation(self):
  m=FakeModel()
  with self.assertRaises(ValueError):checked_load(m,{})
  self.assertEqual(m.calls,0)
 def test_unexpected_rejected(self):
  with self.assertRaises(ValueError):checked_load(FakeModel(),{'weight':Tensor((2,3)),'other':Tensor((1,))})
 def test_shape_rejected(self):
  with self.assertRaises(ValueError):checked_load(FakeModel(),{'weight':Tensor((3,3))})
 def test_no_rope_required_model_allowed(self):checked_load(FakeModel(False),{'weight':Tensor((2,3))})
 def test_rope_unexpected_not_allowed(self):
  with self.assertRaises(ValueError):checked_load(FakeModel(False),{'weight':Tensor((2,3)),'rope_phases':Tensor((1,))})
 def adapter_model(self):
  m=FakeModel();names=[f'blocks.{i//5}.{TARGETS[i%5]}' for i in range(150)];s={}
  from types import SimpleNamespace
  for n in names:
   m.layers[n]=SimpleNamespace(rank=8,alpha=8.,in_features=3,out_features=2)
   s[n+'.lora_down.weight']=Tensor((8,3));s[n+'.lora_up.weight']=Tensor((2,8))
  return m,names,s
 def test_lora150_correct(self):
  m,n,s=self.adapter_model();validate_lora(m,n,s,8,8.,0)
 def test_lora_missing_pair(self):
  m,n,s=self.adapter_model();s.pop(next(iter(s)))
  with self.assertRaises(ValueError):validate_lora(m,n,s,8,8.,0)
 def test_wrong_layer_count(self):
  m,n,s=self.adapter_model()
  with self.assertRaises(ValueError):validate_lora(m,n[:149],s,8,8.,0)
 def test_wrong_rank_and_alpha(self):
  m,n,s=self.adapter_model()
  for r,a in [(4,8),(8,4),(8,float('nan'))]:
   with self.assertRaises(ValueError):validate_lora(m,n,s,r,a,0)
 def test_missing_file(self):
  with self.assertRaises(FileNotFoundError):require_file('/does-not-exist/explicit.pt')
 def test_hash_once(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_bytes(b'123');h=HashCache();x=h.record(p)
   with patch('builtins.open',side_effect=RuntimeError('should not reopen')):self.assertEqual(x,h.record(p))
 def test_rotation_input_unchanged_and_det(self):
  a=np.array([[1.,2.,3.],[4.,5.,6.]]);b=glb_coordinates(a)
  np.testing.assert_array_equal(b,[[1,3,-2],[4,6,-5]]);np.testing.assert_array_equal(a,[[1,2,3],[4,5,6]])
  self.assertAlmostEqual(np.linalg.det(glb_coordinates(np.eye(3))),1)
 def test_raw_not_formal(self):
  self.assertFalse(export_metadata(fast_glb=True)['formal_acceptance_eligible']);self.assertEqual(export_metadata(geometry_only=True)['mode'],'raw_no_remesh');self.assertTrue(export_metadata()['texture_sampling_still_runs'])
 def test_batch_partial_failure_nonzero_manifest(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'out';out.write_bytes(b'123');m={'flags':{'ss_canonicalize':False},'seed':42}
   def run(p,t):
    if p=='bad':raise RuntimeError('bad input')
    return None,str(out)
   (Path(td)/'ok').write_bytes(b'inputgood');(Path(td)/'bad').write_bytes(b'inputbad')
   def actualrun(path,t):return run(Path(path).name,t)
   self.assertEqual(run_batch([str(Path(td)/'ok'),str(Path(td)/'bad')],actualrun,m,Path(td)/'manifest.json'),1)
   self.assertEqual(m['status'],'failed');self.assertEqual(m['successful_images'],1);self.assertEqual(m['failed_images'],1)
   self.assertEqual(m['images'][0]['outputs'][0]['size_bytes'],3);self.assertFalse(m['flags']['ss_canonicalize']);self.assertEqual(m['images'][0]['input_artifact']['sha256'],hashlib.sha256(b'inputgood').hexdigest());self.assertEqual(m['images'][0]['outputs'][0]['sha256'],hashlib.sha256(b'123').hexdigest());self.assertIn('finished_utc',m)
 def test_empty_batch_nonzero(self):
  with tempfile.TemporaryDirectory() as td:self.assertEqual(run_batch([],None,{},Path(td)/'manifest'),1)
 def test_empty_output_failure(self):
  with tempfile.TemporaryDirectory() as td:
   source=Path(td)/'x';source.write_bytes(b'input');self.assertEqual(run_batch([str(source)],lambda p,t:[],{},Path(td)/'manifest'),1)
 def test_discover_preflight(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(ValueError):discover_images(input_dir=td)
   for f in ['x.png','x.jpg']:(Path(td)/f).write_bytes(b'a')
   with self.assertRaises(ValueError):discover_images(input_dir=td)
 def test_pipeline_methods_restored(self):
  class Pipe:
   def sample_sparse_structure(self):return 'ss'
   def sample_shape_slat(self):return 'shape'
   def sample_tex_slat(self):raise RuntimeError('tex fail')
  p=Pipe();r={}
  with pipeline_timing(p,r):
   self.assertEqual(p.sample_sparse_structure(),'ss');p.sample_shape_slat()
   with self.assertRaises(RuntimeError):p.sample_tex_slat()
  self.assertFalse('sample_sparse_structure' in p.__dict__);self.assertEqual(set(r),{'ss','shape','texture'})
 @unittest.skipUnless(hasattr(signal,'setitimer'),'Unix only')
 def test_timeout_interrupts_python(self):
  with self.assertRaises(TimeoutError):
   with image_deadline(.01):time.sleep(.05)
 def test_invalid_timeout(self):
  with self.assertRaises(ValueError):
   with image_deadline(float('nan')):pass
 def test_script_main_preflight_precedes_pipeline(self):
  source=Path('scripts/infer_finetuned.py').read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
  s=ast.get_source_segment(source,node)
  self.assertLess(s.index('require_file(value)'),s.index('Trellis2ImageTo3DPipeline.from_pretrained'))
  self.assertLess(s.index('discover_images('),s.index('pipeline.cuda()'))
 def test_script_geometry_rotated_formal_not_double(self):
  s=Path('scripts/infer_finetuned.py').read_text()
  self.assertEqual(s.count('vertices=glb_coordinates(mesh.vertices.detach().cpu().numpy())'),2)
  self.assertIn('vertices=mesh.vertices, faces=mesh.faces',s)

class MainStubTests(unittest.TestCase):
 def namespace(self):
  import argparse,glob,sys,os
  from types import SimpleNamespace
  import scripts.utils.inference_contract as helper
  namespace={k:getattr(helper,k) for k in dir(helper) if not k.startswith('__')}
  tree=ast.parse(Path('scripts/infer_finetuned.py').read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
  class Pipeline:
   calls=0
   @classmethod
   def from_pretrained(cls,path):cls.calls+=1;return cls()
   def __init__(self):self.models={};self.ss_canonicalize=False;self.ss_swap_yz=False
   def cuda(self):pass
  namespace.update(argparse=argparse,glob=glob,sys=sys,os=os,json=json,__file__=str(Path('scripts/infer_finetuned.py').resolve()),_ACTIVE_RUN=None,
     Trellis2ImageTo3DPipeline=Pipeline,torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)),
     load_finetuned_weights=lambda *a,**k:[],EnvMap=lambda *a,**k:None)
  exec(compile(ast.Module(body=[main],type_ignores=[]),'main_stub','exec'),namespace)
  return namespace,Pipeline
 def test_explicit_weight_fails_before_model(self):
  namespace,pipeline=self.namespace()
  with patch('sys.argv',['infer','--image','missing.png','--sparse_structure_lora','missing.pt']):
   with self.assertRaises(FileNotFoundError):namespace['main']()
  self.assertEqual(pipeline.calls,0)
 def test_missing_image_precedes_model(self):
  namespace,pipeline=self.namespace()
  with patch('sys.argv',['infer','--image','missing.png']):
   with self.assertRaises(FileNotFoundError):namespace['main']()
  self.assertEqual(pipeline.calls,0)
 def test_empty_dir_precedes_model(self):
  namespace,pipeline=self.namespace()
  with tempfile.TemporaryDirectory() as td:
   with patch('sys.argv',['infer','--input_dir',td]):
    with self.assertRaises(ValueError):namespace['main']()
  self.assertEqual(pipeline.calls,0)
 def test_main_partialfailure_manifest_flags(self):
  namespace,pipeline=self.namespace()
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);images=root/'images';images.mkdir();(images/'ok.png').write_bytes(b'a');(images/'bad.png').write_bytes(b'b');checkpoints=root/'checkpoints';checkpoints.mkdir();(checkpoints/'pipeline.json').write_text(json.dumps({'args':{'models':{}}}));out=root/'out'
   def infer(path,pipe,output,**kwargs):
    if Path(path).stem=='bad':raise RuntimeError('mock image failure')
    kwargs['timings']['pipeline_run']=[.1];p=Path(output)/'ok.glb';p.write_bytes(b'output');return None,str(p)
   namespace['infer_single']=infer
   with patch('sys.argv',['infer','--input_dir',str(images),'--output',str(out),'--checkpoints_dir',str(checkpoints),'--pretrained_only','--no_video','--pipeline_type','1024','--seed','42']):self.assertEqual(namespace['main'](),1)
   m=json.loads((out/'run_manifest.json').read_text());self.assertEqual(m['status'],'failed');self.assertEqual(m['failed_images'],1);self.assertFalse(m['flags']['ss_canonicalize']);self.assertFalse(m['flags']['ss_swap_yz']);self.assertEqual(m['pipeline_type'],'1024');self.assertEqual(m['seed'],42);self.assertTrue(m['export']['formal_acceptance_eligible']);self.assertEqual(m['schema'],'trellis.inference-run.v2');self.assertTrue(all('sha256' in x['input_artifact'] for x in m['images']));self.assertIn('created_utc',m);self.assertIn('finished_utc',m)
 def test_main_pretrained_lora_conflict(self):
  namespace,pipeline=self.namespace()
  with tempfile.TemporaryDirectory() as td:
   weight=Path(td)/'x.pt';weight.write_bytes(b'w')
   with patch('sys.argv',['infer','--image','x.png','--pretrained_only','--sparse_structure_lora',str(weight)]):
    with self.assertRaises(SystemExit):namespace['main']()
  self.assertEqual(pipeline.calls,0)

if __name__=='__main__':unittest.main(verbosity=2)
