import unittest
import torch
from trellis2.modules.lora import LoRALinear,apply_lora_to_linear_layers

class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.target=torch.nn.Linear(4,3);self.other=torch.nn.Linear(3,2)

class LoraIdempotenceTests(unittest.TestCase):
 def test_repeat_preserves_identity_keys_values_and_output(self):
  torch.manual_seed(5);m=Tiny();x=torch.randn(2,4);base=m.target(x).detach();first=apply_lora_to_linear_layers(m,['target'],2,2);self.assertEqual(first,['target']);torch.testing.assert_close(m.target(x),base,rtol=0,atol=0)
  with torch.no_grad():m.target.lora_up.weight.normal_()
  before=m.target(x).detach();keys=set(m.state_dict());refs={k:p for k,p in m.named_parameters()};values={k:p.detach().clone() for k,p in refs.items()}
  self.assertEqual(apply_lora_to_linear_layers(m,['target'],2,2),[]);self.assertEqual(set(m.state_dict()),keys)
  for k,p in m.named_parameters():self.assertIs(p,refs[k]);self.assertTrue(torch.equal(p,values[k]))
  self.assertTrue(torch.equal(m.target(x),before))
 def test_internal_keyword_never_wraps_existing_adapter(self):
  m=Tiny();apply_lora_to_linear_layers(m,['target'],2,2);self.assertEqual(apply_lora_to_linear_layers(m,['lora_down','lora_up'],2,2),[]);self.assertIsInstance(m.target.lora_down,torch.nn.Linear)
 def test_new_unwrapped_target_can_still_be_injected(self):
  m=Tiny();apply_lora_to_linear_layers(m,['target'],2,2);self.assertEqual(apply_lora_to_linear_layers(m,['target','other'],2,2),['other']);self.assertEqual(sum(isinstance(v,LoRALinear) for v in m.modules()),2)
 def test_nested_sequential_repeat(self):
  m=torch.nn.Sequential(Tiny(),Tiny());self.assertEqual(len(apply_lora_to_linear_layers(m,['target'],2,2)),2);self.assertEqual(apply_lora_to_linear_layers(m,['target'],2,2),[]);self.assertEqual(sum(isinstance(v,LoRALinear) for v in m.modules()),2)
 def test_root_wrapped_layer_unchanged(self):
  m=LoRALinear(torch.nn.Linear(4,3),2,2);self.assertEqual(apply_lora_to_linear_layers(m,['lora'],2,2),[]);self.assertIsInstance(m.lora_down,torch.nn.Linear)
if __name__=='__main__':unittest.main(verbosity=2)
