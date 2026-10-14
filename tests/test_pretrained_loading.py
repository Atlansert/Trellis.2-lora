"""预训练权重加载契约的 CPU 单元测试（models.from_pretrained / Pipeline.from_pretrained）。

- 覆盖点：权重严格校验（键缺失或多余都抛 CheckpointValidationError）；rope_phases 等 buffer
  由代码重新生成而不是照搬 checkpoint（dtype/数值以代码为准）；本地路径不完整时抛 FileNotFoundError
  且绝不联网回退；pipeline.json 里引用的本地组件缺失或损坏时不得回退到远端（loader 不被调用或只调一次）；
  跨仓库引用（org/repo/...）才允许直接加载。
- 运行：python -m pytest tests/test_pretrained_loading.py（纯 CPU，用 tiny 假模型 + mock，不下载权重）。
- 依赖：torch、safetensors、trellis2.models、trellis2.pipelines.base.Pipeline。
"""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
import trellis2.models as models
from trellis2.pipelines.base import Pipeline

# 被替换进来的极简模型：一个权重参数 + 一个复数 buffer（用于验证 buffer 由代码重算）。
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2));self.register_buffer('rope_phases',torch.tensor([1j],dtype=torch.complex64))

class PretrainedLoadingTests(unittest.TestCase):
 # 造一个本地“模型目录”：tiny.json（配置）+ tiny.safetensors（权重），返回无后缀路径。
 def files(self,td,state):
  p=Path(td)/'tiny';p.with_suffix('.json').write_text(json.dumps({'name':'Tiny','args':{}}));save_file(state,str(p.with_suffix('.safetensors')));return p
 # 契约：权重按 checkpoint 装载（weight=1），但 rope_phases 必须由代码寄存器重建（1j/complex64），
 # 并记录 _checkpoint_source['model_file'] 便于溯源。
 def test_exact_weights_and_regenerated_rope(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'weight':torch.ones(2),'rope_phases':torch.zeros(1)})
   with patch.object(models,'__getattr__',return_value=Tiny):m=models.from_pretrained(str(p))
   self.assertTrue(m.weight.eq(1).all());self.assertEqual(m.rope_phases.dtype,torch.complex64);self.assertEqual(m.rope_phases.item(),1j);self.assertEqual(m._checkpoint_source['model_file'],str(p.with_suffix('.safetensors')))
 # 契约：缺少必需权重必须抛 CheckpointValidationError（不允许静默跳过）。
 def test_missing_weight_fails(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'other':torch.ones(2)})
   with patch.object(models,'__getattr__',return_value=Tiny):
    with self.assertRaises(models.CheckpointValidationError):models.from_pretrained(str(p))
 # 契约：checkpoint 里有模型用不到的键也必须报错（严格加载）。
 def test_unexpected_weight_fails(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'weight':torch.ones(2),'other':torch.ones(1)})
   with patch.object(models,'__getattr__',return_value=Tiny):
    with self.assertRaises(models.CheckpointValidationError):models.from_pretrained(str(p))
 # 契约：本地目录缺少 .safetensors 时直接 FileNotFoundError，不得尝试联网下载。
 def test_incomplete_local_path_no_network(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'tiny';p.with_suffix('.json').write_text('{}')
   with self.assertRaises(FileNotFoundError):models.from_pretrained(str(p))
 # 契约：pipeline 引用的本地组件路径不存在时抛 FileNotFoundError，且不得调用加载器去回退。
 def test_missing_local_pipeline_component_no_fallback(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/'pipeline.json').write_text(json.dumps({'args':{'models':{'tiny':'ckpts/missing'}}}))
   with patch.object(models,'from_pretrained') as loader:
    with self.assertRaises(FileNotFoundError):Pipeline.from_pretrained(td)
    loader.assert_not_called()
 # 契约：本地组件校验失败时异常原样抛出，加载器只被调用一次（不做任何回退重试）。
 def test_broken_local_component_never_falls_back(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'weight':torch.ones(2)});(Path(td)/'pipeline.json').write_text(json.dumps({'args':{'models':{'tiny':'tiny'}}}))
   with patch.object(models,'from_pretrained',side_effect=models.CheckpointValidationError('bad')) as loader:
    with self.assertRaises(models.CheckpointValidationError):Pipeline.from_pretrained(td)
    self.assertEqual(loader.call_count,1)
 # 契约：跨仓库引用（形如 org/repo/ckpts/model，本地不存在）允许按原字符串直接加载。
 def test_cross_repo_component_is_loaded_directly(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/'pipeline.json').write_text(json.dumps({'args':{'models':{'tiny':'org/repo/ckpts/model'}}}))
   with patch.object(models,'from_pretrained',return_value=Tiny()) as loader:Pipeline.from_pretrained(td)
   loader.assert_called_once_with('org/repo/ckpts/model')
if __name__=='__main__':unittest.main(verbosity=2)
