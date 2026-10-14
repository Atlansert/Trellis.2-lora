import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
import trellis2.models as models
from trellis2.pipelines.base import Pipeline

class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2));self.register_buffer('rope_phases',torch.tensor([1j],dtype=torch.complex64))

class PretrainedLoadingTests(unittest.TestCase):
 def files(self,td,state):
  p=Path(td)/'tiny';p.with_suffix('.json').write_text(json.dumps({'name':'Tiny','args':{}}));save_file(state,str(p.with_suffix('.safetensors')));return p
 def test_exact_weights_and_regenerated_rope(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'weight':torch.ones(2),'rope_phases':torch.zeros(1)})
   with patch.object(models,'__getattr__',return_value=Tiny):m=models.from_pretrained(str(p))
   self.assertTrue(m.weight.eq(1).all());self.assertEqual(m.rope_phases.dtype,torch.complex64);self.assertEqual(m.rope_phases.item(),1j);self.assertEqual(m._checkpoint_source['model_file'],str(p.with_suffix('.safetensors')))
 def test_missing_weight_fails(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'other':torch.ones(2)})
   with patch.object(models,'__getattr__',return_value=Tiny):
    with self.assertRaises(models.CheckpointValidationError):models.from_pretrained(str(p))
 def test_unexpected_weight_fails(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'weight':torch.ones(2),'other':torch.ones(1)})
   with patch.object(models,'__getattr__',return_value=Tiny):
    with self.assertRaises(models.CheckpointValidationError):models.from_pretrained(str(p))
 def test_incomplete_local_path_no_network(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'tiny';p.with_suffix('.json').write_text('{}')
   with self.assertRaises(FileNotFoundError):models.from_pretrained(str(p))
 def test_missing_local_pipeline_component_no_fallback(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/'pipeline.json').write_text(json.dumps({'args':{'models':{'tiny':'ckpts/missing'}}}))
   with patch.object(models,'from_pretrained') as loader:
    with self.assertRaises(FileNotFoundError):Pipeline.from_pretrained(td)
    loader.assert_not_called()
 def test_broken_local_component_never_falls_back(self):
  with tempfile.TemporaryDirectory() as td:
   p=self.files(td,{'weight':torch.ones(2)});(Path(td)/'pipeline.json').write_text(json.dumps({'args':{'models':{'tiny':'tiny'}}}))
   with patch.object(models,'from_pretrained',side_effect=models.CheckpointValidationError('bad')) as loader:
    with self.assertRaises(models.CheckpointValidationError):Pipeline.from_pretrained(td)
    self.assertEqual(loader.call_count,1)
 def test_cross_repo_component_is_loaded_directly(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/'pipeline.json').write_text(json.dumps({'args':{'models':{'tiny':'org/repo/ckpts/model'}}}))
   with patch.object(models,'from_pretrained',return_value=Tiny()) as loader:Pipeline.from_pretrained(td)
   loader.assert_called_once_with('org/repo/ckpts/model')
if __name__=='__main__':unittest.main(verbosity=2)
