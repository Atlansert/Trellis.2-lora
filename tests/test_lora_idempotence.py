"""LoRA 注入的幂等性 CPU 单元测试（apply_lora_to_linear_layers）。

- 覆盖点：首次注入后输出与注入前完全一致（零初始化，atol=0）；重复调用返回空列表且不改变
  state_dict 键、参数对象身份与数值；内部关键字（lora_down/lora_up）不会被二次包装；
  已包装模块里未包装的其它层仍可注入；嵌套 Sequential 下重复调用仍幂等；根节点本身是 LoRALinear 时不再包装。
- 运行：python -m pytest tests/test_lora_idempotence.py（纯 CPU，仅用随机小张量，不需要数据/权重）。
- 依赖：torch 与 trellis2.modules.lora。
"""
import unittest
import torch
from trellis2.modules.lora import LoRALinear,apply_lora_to_linear_layers

# 最小被测模块：一个目标 Linear + 一个不相关 Linear。
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.target=torch.nn.Linear(4,3);self.other=torch.nn.Linear(3,2)

class LoraIdempotenceTests(unittest.TestCase):
 # 契约：重复注入必须完全幂等——不新增键、不换参数对象、不改数值，且破坏性改动 lora_up 后
 # 重复注入也不能把已有 adapter 重置（输出与改动后一致）。
 def test_repeat_preserves_identity_keys_values_and_output(self):
  torch.manual_seed(5);m=Tiny();x=torch.randn(2,4);base=m.target(x).detach();first=apply_lora_to_linear_layers(m,['target'],2,2);self.assertEqual(first,['target']);torch.testing.assert_close(m.target(x),base,rtol=0,atol=0)
  with torch.no_grad():m.target.lora_up.weight.normal_()
  before=m.target(x).detach();keys=set(m.state_dict());refs={k:p for k,p in m.named_parameters()};values={k:p.detach().clone() for k,p in refs.items()}
  self.assertEqual(apply_lora_to_linear_layers(m,['target'],2,2),[]);self.assertEqual(set(m.state_dict()),keys)
  for k,p in m.named_parameters():self.assertIs(p,refs[k]);self.assertTrue(torch.equal(p,values[k]))
  self.assertTrue(torch.equal(m.target(x),before))
 # 契约：把内部参数名当目标传入时不得包装已存在的 adapter。
 def test_internal_keyword_never_wraps_existing_adapter(self):
  m=Tiny();apply_lora_to_linear_layers(m,['target'],2,2);self.assertEqual(apply_lora_to_linear_layers(m,['lora_down','lora_up'],2,2),[]);self.assertIsInstance(m.target.lora_down,torch.nn.Linear)
 # 契约：只对尚未包装的层生效，返回本次真正注入的名字列表。
 def test_new_unwrapped_target_can_still_be_injected(self):
  m=Tiny();apply_lora_to_linear_layers(m,['target'],2,2);self.assertEqual(apply_lora_to_linear_layers(m,['target','other'],2,2),['other']);self.assertEqual(sum(isinstance(v,LoRALinear) for v in m.modules()),2)
 # 契约：嵌套容器里同名子模块各自注入一次，重复调用仍幂等（总数不变）。
 def test_nested_sequential_repeat(self):
  m=torch.nn.Sequential(Tiny(),Tiny());self.assertEqual(len(apply_lora_to_linear_layers(m,['target'],2,2)),2);self.assertEqual(apply_lora_to_linear_layers(m,['target'],2,2),[]);self.assertEqual(sum(isinstance(v,LoRALinear) for v in m.modules()),2)
 # 契约：根节点本身已是 LoRALinear 时不再包装（避免套娃）。
 def test_root_wrapped_layer_unchanged(self):
  m=LoRALinear(torch.nn.Linear(4,3),2,2);self.assertEqual(apply_lora_to_linear_layers(m,['lora'],2,2),[]);self.assertIsInstance(m.lora_down,torch.nn.Linear)
if __name__=='__main__':unittest.main(verbosity=2)
