"""推理契约（scripts/utils/inference_contract.py 与 scripts/infer_finetuned.py）的 CPU 单元测试。

- 覆盖点：权重严格加载（缺失/多余/形状不符都拒绝，且必须在改动模型前失败）、rope_phases 缺失可容忍
  但多余不行、LoRA adapter 的数量/配对/rank/alpha 校验、文件哈希缓存（同一文件只读一次）、
  glb_coordinates 轴变换为纯旋转（det=+1）且不改动输入、导出元数据的 raw/formal 判定、
  批量推理部分失败仍写 manifest 且退出码非 0、空批次/空输出失败、输入图片预检、
  pipeline 计时包装的方法还原、Unix 超时中断，以及用 AST 从 infer_finetuned.py 抽取 main() 验证
  “预检先于建模型/上 GPU”与 glb 坐标只旋转一次（不重复旋转）。
- 运行：python -m pytest tests/test_inference_contract.py（纯 CPU；用假模型与 mock，不加载真实权重）。
- 依赖：numpy、scripts.utils.inference_contract、scripts/infer_finetuned.py（按相对路径读取源码）。
"""
import ast,contextlib,io,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.utils.inference_contract import *

# 假的张量：只需要 shape 属性即可满足形状校验。
class Tensor:
 def __init__(self,shape):self.shape=shape
# 假模型：带 state_dict/load_state_dict/eval/get_submodule，用于校验加载前的检查而不真的载入权重。
class FakeModel:
 def __init__(self,rope=True):self.state={'weight':Tensor((2,3))};self.calls=0;self.training=True;self.layers={};self.state.update({'rope_phases':Tensor((1,))} if rope else {})
 def state_dict(self):return self.state
 def load_state_dict(self,state,strict=False):self.calls+=1;return list(set(self.state)-set(state)),list(set(state)-set(self.state))
 def eval(self):self.training=False
 def get_submodule(self,name):return self.layers[name]
class ContractTests(unittest.TestCase):
 # 契约：rope_phases 缺失是允许的（由代码重算），加载后模型必须切到 eval 模式。
 def test_rope_missing_allowed_and_eval(self):
  m=FakeModel();checked_load(m,{'weight':Tensor((2,3))});self.assertFalse(m.training)
 # 契约：缺少必需权重时必须在调用 load_state_dict 之前失败（calls 仍为 0，模型不被改动）。
 def test_bad_missing_rejected_before_mutation(self):
  m=FakeModel()
  with self.assertRaises(ValueError):checked_load(m,{})
  self.assertEqual(m.calls,0)
 # 契约：checkpoint 里出现未知键必须拒绝（严格加载）。
 def test_unexpected_rejected(self):
  with self.assertRaises(ValueError):checked_load(FakeModel(),{'weight':Tensor((2,3)),'other':Tensor((1,))})
 # 契约：形状不匹配（(3,3) vs (2,3)）必须拒绝。
 def test_shape_rejected(self):
  with self.assertRaises(ValueError):checked_load(FakeModel(),{'weight':Tensor((3,3))})
 # 契约：模型本身不需要 rope_phases 时，缺失该键当然也允许加载。
 def test_no_rope_required_model_allowed(self):checked_load(FakeModel(False),{'weight':Tensor((2,3))})
 # 契约：模型不需要 rope 时，checkpoint 里多余的 rope_phases 反而必须拒绝。
 def test_rope_unexpected_not_allowed(self):
  with self.assertRaises(ValueError):checked_load(FakeModel(False),{'weight':Tensor((2,3)),'rope_phases':Tensor((1,))})
 # 造 150 个 LoRA 层（每层 rank=8/alpha=8，in=3/out=2）及其权重字典，用于容量与配对校验。
 def adapter_model(self):
  m=FakeModel();names=[f'blocks.{i//5}.{TARGETS[i%5]}' for i in range(150)];s={}
  from types import SimpleNamespace
  for n in names:
   m.layers[n]=SimpleNamespace(rank=8,alpha=8.,in_features=3,out_features=2)
   s[n+'.lora_down.weight']=Tensor((8,3));s[n+'.lora_up.weight']=Tensor((2,8))
  return m,names,s
 # 契约：150 层 LoRA 全部合法时校验通过（覆盖层数上限这一档）。
 def test_lora150_correct(self):
  m,n,s=self.adapter_model();validate_lora(m,n,s,8,8.,0)
 # 契约：lora_down/lora_up 必须成对出现，缺一个就拒绝。
 def test_lora_missing_pair(self):
  m,n,s=self.adapter_model();s.pop(next(iter(s)))
  with self.assertRaises(ValueError):validate_lora(m,n,s,8,8.,0)
 # 契约：目标层数与实际注入层数不一致（149 vs 150）必须拒绝。
 def test_wrong_layer_count(self):
  m,n,s=self.adapter_model()
  with self.assertRaises(ValueError):validate_lora(m,n[:149],s,8,8.,0)
 # 契约：rank/alpha 与契约不符（含 NaN alpha）都要拒绝。
 def test_wrong_rank_and_alpha(self):
  m,n,s=self.adapter_model()
  for r,a in [(4,8),(8,4),(8,float('nan'))]:
   with self.assertRaises(ValueError):validate_lora(m,n,s,r,a,0)
 # 契约：显式指定的权重文件不存在时抛 FileNotFoundError（不做静默回退）。
 def test_missing_file(self):
  with self.assertRaises(FileNotFoundError):require_file('/does-not-exist/explicit.pt')
 # 契约：同一文件哈希只计算一次（第二次命中缓存，不再打开文件）。
 def test_hash_once(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_bytes(b'123');h=HashCache();x=h.record(p)
   with patch('builtins.open',side_effect=RuntimeError('should not reopen')):self.assertEqual(x,h.record(p))
 # 契约：正式 GLB 坐标变换 (x,z,-y) 是纯旋转（det=+1）且不改动输入数组。
 def test_rotation_input_unchanged_and_det(self):
  a=np.array([[1.,2.,3.],[4.,5.,6.]]);b=glb_coordinates(a)
  np.testing.assert_array_equal(b,[[1,3,-2],[4,6,-5]]);np.testing.assert_array_equal(a,[[1,2,3],[4,5,6]])
  self.assertAlmostEqual(np.linalg.det(glb_coordinates(np.eye(3))),1)
 # 契约：fast_glb/geometry_only 产出的都是 raw（不可用于正式验收），只有默认路径才算正式导出。
 def test_raw_not_formal(self):
  self.assertFalse(export_metadata(fast_glb=True)['formal_acceptance_eligible']);self.assertEqual(export_metadata(geometry_only=True)['mode'],'raw_no_remesh');self.assertTrue(export_metadata()['texture_sampling_still_runs'])
 # 契约（核心）：一张成功一张失败时 run_batch 返回非 0，manifest 记 status=failed、
 # 成功/失败计数、逐图输入输出哈希与大小、原始 flags 保留，并写入 finished_utc。
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
 # 契约：空批次必须返回非 0（不把“什么都没做”当成功）。
 def test_empty_batch_nonzero(self):
  with tempfile.TemporaryDirectory() as td:self.assertEqual(run_batch([],None,{},Path(td)/'manifest'),1)
 # 契约：单图推理没有产出任何文件时也算失败（返回非 0）。
 def test_empty_output_failure(self):
  with tempfile.TemporaryDirectory() as td:
   source=Path(td)/'x';source.write_bytes(b'input');self.assertEqual(run_batch([str(source)],lambda p,t:[],{},Path(td)/'manifest'),1)
 # 契约：输入目录里没有图片、或同时存在多种图像格式（歧义）都必须在开跑前报错。
 def test_discover_preflight(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(ValueError):discover_images(input_dir=td)
   for f in ['x.png','x.jpg']:(Path(td)/f).write_bytes(b'a')
   with self.assertRaises(ValueError):discover_images(input_dir=td)
 # 契约：计时包装只记录 ss/shape/texture 三段耗时，退出后 pipeline 上不得残留被替换的方法，
 # 且即使中间抛错也要还原。
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
 # 契约：图片级超时（signal.setitimer，仅 Unix）能把阻塞的 Python 代码中断成 TimeoutError。
 @unittest.skipUnless(hasattr(signal,'setitimer'),'Unix only')
 def test_timeout_interrupts_python(self):
  with self.assertRaises(TimeoutError):
   with image_deadline(.01):time.sleep(.05)
 # 契约：非有限（NaN）超时值必须拒绝，不允许当作“无超时”绕过。
 def test_invalid_timeout(self):
  with self.assertRaises(ValueError):
   with image_deadline(float('nan')):pass
 # 契约（源码级）：main() 里的文件/图片预检必须出现在建 pipeline 与 .cuda() 之前
 # （按源码文本顺序断言，防止“先上 GPU 再发现输入缺失”）。
 def test_script_main_preflight_precedes_pipeline(self):
  source=Path('scripts/infer_finetuned.py').read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
  s=ast.get_source_segment(source,node)
  self.assertLess(s.index('require_file(value)'),s.index('Trellis2ImageTo3DPipeline.from_pretrained'))
  self.assertLess(s.index('discover_images('),s.index('pipeline.cuda()'))
 # 契约（源码级）：几何坐标只做一次 (x,z,-y) 旋转（恰好 2 处），并保留未经旋转的
 # 'vertices=mesh.vertices, faces=mesh.faces' 用法，避免重复旋转。
 def test_script_geometry_rotated_formal_not_double(self):
  s=Path('scripts/infer_finetuned.py').read_text()
  self.assertEqual(s.count('vertices=glb_coordinates(mesh.vertices.detach().cpu().numpy())'),2)
  self.assertIn('vertices=mesh.vertices, faces=mesh.faces',s)

class MainStubTests(unittest.TestCase):
 # 把 infer_finetuned.py 的 main() 单独 exec 进受控命名空间：只替换 pipeline/torch 等外围依赖，
 # 从而在 CPU 上跑真实的 main() 流程（含参数解析与 manifest 写出）。
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
 # 契约：显式 --sparse_structure_lora 文件不存在时，必须在建 pipeline 之前就失败。
 def test_explicit_weight_fails_before_model(self):
  namespace,pipeline=self.namespace()
  with patch('sys.argv',['infer','--image','missing.png','--sparse_structure_lora','missing.pt']):
   with self.assertRaises(FileNotFoundError):namespace['main']()
  self.assertEqual(pipeline.calls,0)
 # 契约：输入图片不存在也要在建 pipeline 之前失败。
 def test_missing_image_precedes_model(self):
  namespace,pipeline=self.namespace()
  with patch('sys.argv',['infer','--image','missing.png']):
   with self.assertRaises(FileNotFoundError):namespace['main']()
  self.assertEqual(pipeline.calls,0)
 # 契约：输入目录里没有可用图片时同样在建 pipeline 之前失败。
 def test_empty_dir_precedes_model(self):
  namespace,pipeline=self.namespace()
  with tempfile.TemporaryDirectory() as td:
   with patch('sys.argv',['infer','--input_dir',td]):
    with self.assertRaises(ValueError):namespace['main']()
  self.assertEqual(pipeline.calls,0)
 # 契约（核心）：真实 main() 在部分图片失败时返回 1，并写出 v2 manifest——status=failed、
 # 失败计数、改轴标志全为 False、pipeline_type/seed 如实记录、导出为正式（可验收）、
 # 每张图都有输入哈希，且含 created_utc/finished_utc。
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
 # 契约：--pretrained_only 与 --sparse_structure_lora 互斥，参数冲突以 SystemExit 结束且不建模型。
 def test_main_pretrained_lora_conflict(self):
  namespace,pipeline=self.namespace()
  with tempfile.TemporaryDirectory() as td:
   weight=Path(td)/'x.pt';weight.write_bytes(b'w')
   with patch('sys.argv',['infer','--image','x.png','--pretrained_only','--sparse_structure_lora',str(weight)]):
    with self.assertRaises(SystemExit):namespace['main']()
  self.assertEqual(pipeline.calls,0)

if __name__=='__main__':unittest.main(verbosity=2)
