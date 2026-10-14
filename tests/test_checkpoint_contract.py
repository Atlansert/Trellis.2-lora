"""检查点契约（trellis2.utils.checkpoint_contract）的 CPU 单元测试。

- 覆盖点：原子保存与 marker、完整步骤枚举、哈希/大小校验、覆盖拒绝（FileExistsError）、
  损坏 marker 不得静默取最新、RNG 状态的保存与恢复（含非 rank0 与旧版缺失的告警）、
  LoRA adapter 与复数 buffer 在 dtype 转换中的精确保留、optimizer 动量的 fp32 对齐、
  以及用 AST 抽取的**真实 BasicTrainer.load/save/finetune_from/run 分支**做端到端一致性验证。
- 运行：python -m pytest tests/test_checkpoint_contract.py（纯 CPU：用 bfloat16/CPU autocast 与
  tiny 模型，不需要 GPU、数据或真实检查点；临时目录内的改动会自行清理）。
- 依赖：torch、numpy、trellis2.modules.lora、trellis2.utils.checkpoint_contract，
  并从 trellis2/trainers/basic.py **按源码读入**方法体（因此本测试要求在这些路径下运行）。
"""
import ast, contextlib, copy, io, json, os, random, tempfile, types, unittest, warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from trellis2.modules.lora import LoRALinear
from trellis2.utils import checkpoint_contract as cc

# Exercise exact live BasicTrainer methods without importing CUDA trainer setup.
# 中文：从 trellis2/trainers/basic.py 源码里抽出真实方法体，在一个最小命名空间里 exec 成 TinyBasic，
# 这样既能测真实实现，又不必导入需要 CUDA 的训练器初始化。
source=Path('trellis2/trainers/basic.py').read_text()
module=ast.parse(source);klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='BasicTrainer')
names=['load','save','finetune_from','_master_params_to_state_dicts','_state_dicts_to_master_params']
methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in names]
ns=dict(vars(cc));ns.update(torch=torch,os=os,warnings=warnings,read_file_dist=lambda p:p)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='TinyBasic',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),'live-basic-methods','exec'),ns)
TinyBasic=ns['TinyBasic']

# 被测假模型：含 LoRA 线性层 + 两个复数 buffer（用于验证“只把 LoRA 转 fp32、复数 buffer 保留”）。
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.proj=LoRALinear(torch.nn.Linear(3,2,dtype=torch.bfloat16),rank=2,alpha=2)
  self.register_buffer('rope_phases',torch.tensor([1j],dtype=torch.complex64))
  self.register_buffer('other_complex',torch.tensor([2j],dtype=torch.complex64))
 def forward(self,x):return self.proj(x)

class Sampler:
 def __init__(self):self.state={'position':7}
 def state_dict(self):return self.state
 def load_state_dict(self,s):self.state=s

# 造一个最小可用的 TinyBasic 实例（单 rank、CPU、amp/bfloat16，含 ema、optimizer、data_sampler）。
def tiny(td,model=None):
 t=TinyBasic();t.models={'denoiser':model or Model()};t.master_params=[p for p in t.models['denoiser'].parameters() if p.requires_grad];t.model_params=t.master_params
 t.ema_rate=[.9999];t.ema_params=[copy.deepcopy(t.master_params)];t.optimizer=torch.optim.AdamW(t.master_params,lr=.01)
 t.is_master=True;t.rank=0;t.world_size=1;t.device=torch.device('cpu');t.mix_precision_mode='amp';t.mix_precision_dtype=torch.bfloat16;t.output_dir=td;t.step=3;t.data_sampler=Sampler()
 t.lr_scheduler_config=None;t.elastic_controller_config=None;t.grad_clip=None
 return t

# 走一步优化：用于验证“保存/加载后下一步的 loss 与参数完全一致”。
def update(t):
 t.optimizer.zero_grad();x=torch.tensor([[.4,.2,-.3]],dtype=torch.bfloat16)
 with torch.autocast('cpu',dtype=torch.bfloat16):loss=t.models['denoiser'](x).float().square().sum()
 loss.backward();t.optimizer.step();return loss.detach()

# 固定的保存载荷：denoiser/misc 两个文件（文件名里的 step 必须与传入 step 一致）。
def payload(step=3):return {f'denoiser_step{step:07d}.pt':{'x':torch.tensor([1.25])},f'misc_step{step:07d}.pt':{'step':step}}

class CheckpointTests(unittest.TestCase):
 # 契约：atomic_save_group 总是同步完成（non_blocking 只是请求），marker 记录与校验一致。
 def test_atomic_complete_files_hashes_and_sync(self):
  with tempfile.TemporaryDirectory() as td:
   m=cc.atomic_save_group(td,3,payload(),non_blocking=True)
   self.assertTrue(m['synchronous']);self.assertTrue(m['requested_non_blocking']);self.assertEqual(cc.list_complete_steps(td),[3]);self.assertEqual(cc.validate_checkpoint(td,3),m)
 # 契约：保存失败时不得留下 marker（否则会被误认为完整检查点）。
 def test_no_marker_for_save_failure(self):
  with tempfile.TemporaryDirectory() as td:
   with patch.object(cc.torch,'save',side_effect=OSError('disk full')):
    with self.assertRaises(OSError):cc.atomic_save_group(td,3,payload())
   self.assertEqual(cc.list_complete_steps(td),[]);self.assertFalse((Path(td)/cc.marker_name(3)).exists())
 # 契约：写到一半失败时允许留下部分文件，但绝不能出现完整步骤（marker 缺失）。
 def test_failure_after_first_file_keeps_partial_no_complete(self):
  with tempfile.TemporaryDirectory() as td:
   real=torch.save;calls=[]
   def save(v,f):
    calls.append(1)
    if len(calls)==2:raise OSError('second write failed')
    real(v,f)
   with patch.object(cc.torch,'save',side_effect=save):
    with self.assertRaises(OSError):cc.atomic_save_group(td,3,payload())
   self.assertTrue((Path(td)/'denoiser_step0000003.pt').exists());self.assertEqual(cc.list_complete_steps(td),[])
 # 契约：同一 step 已有完整检查点时拒绝覆盖，且原 marker 字节不变。
 def test_complete_overwrite_refused_unchanged(self):
  with tempfile.TemporaryDirectory() as td:
   cc.atomic_save_group(td,3,payload());before=(Path(td)/cc.marker_name(3)).read_bytes()
   with self.assertRaises(FileExistsError):cc.atomic_save_group(td,3,payload())
   self.assertEqual(before,(Path(td)/cc.marker_name(3)).read_bytes())
 # 契约：目标文件已存在（哪怕是残留的部分文件）也拒绝覆盖，原内容保持不变。
 def test_partial_target_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'denoiser_step0000003.pt';p.write_bytes(b'old')
   with self.assertRaises(FileExistsError):cc.atomic_save_group(td,3,payload())
   self.assertEqual(p.read_bytes(),b'old')
 # 契约：marker 损坏时必须报错，不得静默“取最新可用”继续训练。
 def test_bad_marker_raises_not_silent_latest(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/cc.marker_name(3)).write_text('{}')
   with self.assertRaises(ValueError):cc.list_complete_steps(td)
 # 契约：空载荷、只有 misc、路径穿越（'../'）、step 前缀不一致，四种非法载荷都要拒绝。
 def test_empty_misconly_unsafe_wrong_step_refused(self):
  with tempfile.TemporaryDirectory() as td:
   for p in [{},{'misc_step0000003.pt':{}},{'../x_step0000003.pt':{},'misc_step0000003.pt':{}},{'x_step0000004.pt':{},'misc_step0000003.pt':{}}]:
    with self.assertRaises(ValueError):cc.atomic_save_group(td,3,p)
 # 契约：同长度篡改（翻转最后 1 bit）不触发大小检查，但开启哈希校验时必须被发现。
 def test_tamper_same_size_hash_failure(self):
  with tempfile.TemporaryDirectory() as td:
   cc.atomic_save_group(td,3,payload());p=Path(td)/'denoiser_step0000003.pt';b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
   self.assertEqual(cc.list_complete_steps(td,False),[3])
   with self.assertRaises(ValueError):cc.validate_checkpoint(td,3,True)
 # 契约：文件大小与 marker 记录不一致（或文件缺失）即视为不完整。
 def test_missing_size_mismatch(self):
  with tempfile.TemporaryDirectory() as td:
   cc.atomic_save_group(td,3,payload());(Path(td)/'misc_step0000003.pt').write_bytes(b'x')
   with self.assertRaises(ValueError):cc.list_complete_steps(td)
 # 契约：没有 marker 的旧格式文件不算完整步骤（需显式 legacy 兼容路径）。
 def test_legacy_files_not_complete(self):
  with tempfile.TemporaryDirectory() as td:
   torch.save({},Path(td)/'misc_step0000003.pt');self.assertEqual(cc.list_complete_steps(td),[])
 # 契约：快照必须是 CPU 上独立存储的副本，后续改动源张量不影响快照。
 def test_snapshot_owns_cpu_storage(self):
  t=torch.ones(2);s=cc.cpu_snapshot({'x':t});t.fill_(8);self.assertTrue(s['x'].eq(1).all())
 # 契约：python/numpy/torch 三套 RNG 状态都要能完整往返（恢复后取随机数结果一致）。
 def test_rng_roundtrip_python_numpy_torch(self):
  random.seed(17);np.random.seed(17);torch.manual_seed(17);s=cc.capture_rng_state();a=(random.random(),np.random.rand(),torch.rand(2));cc.restore_rng_state(s);b=(random.random(),np.random.rand(),torch.rand(2));self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))
 # 契约：非 rank0 不得套用 rank0 的 RNG 状态（应告警并保持自身 RNG 不变）。
 def test_other_rank_does_not_receive_rank0_rng(self):
  s=cc.capture_rng_state();torch.manual_seed(23);before=torch.get_rng_state().clone()
  with self.assertWarns(RuntimeWarning):self.assertFalse(cc.restore_rng_state(s,rank=1,world_size=2))
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
 # 契约：旧版检查点没有 RNG 状态时必须告警而不是报错。
 def test_legacy_rng_warn(self):
  with self.assertWarns(RuntimeWarning):self.assertFalse(cc.restore_rng_state(None))
 # 契约：dtype 转换后 LoRA adapter 仍是同一对象、数值精确不变且强制 fp32；复数 buffer 保持复数原值。
 def test_cast_exact_adapters_identity_and_all_complex(self):
  m=Model();p=m.proj.lora_down.weight;value=p.detach().clone();complex0=m.other_complex
  cc.cast_model_preserving_lora_and_complex(m,torch.bfloat16)
  self.assertIs(m.proj.lora_down.weight,p);self.assertTrue(torch.equal(p,value));self.assertEqual(p.dtype,torch.float32);self.assertEqual(m.proj.weight.dtype,torch.bfloat16);self.assertIs(m.other_complex,complex0);self.assertEqual(m.rope_phases.item(),1j)
 # 契约：不含 LoRA 的普通模型不应被强制转 fp32（保持目标 dtype）。
 def test_nonlora_model_not_forced_fp32(self):
  m=torch.nn.Linear(3,2);cc.cast_model_preserving_lora_and_complex(m,torch.bfloat16);self.assertEqual(m.weight.dtype,torch.bfloat16)
 # 契约（核心）：真实 save→load 后 master_params 对象身份/数值/fp32 一致、复数 buffer 还原、
 # data_sampler 状态还原，且“下一步”的 loss 与参数更新结果与未中断的训练完全相同。
 def test_live_basic_save_load_next_step_equivalent(self):
  with tempfile.TemporaryDirectory() as td:
   torch.manual_seed(4);a=tiny(td);update(a);a.save(non_blocking=True);b=tiny(td);refs=list(b.master_params);b.load(td,3)
   self.assertEqual(b.step,3);self.assertEqual(b.data_sampler.state,{'position':7})
   for pa,pb,ref in zip(a.master_params,b.master_params,refs):
    self.assertIs(pb,ref);self.assertEqual(pb.dtype,torch.float32);self.assertTrue(torch.equal(pa,pb));self.assertEqual(b.optimizer.state[pb]['exp_avg'].dtype,torch.float32)
   self.assertEqual(b.models['denoiser'].rope_phases.item(),1j);self.assertEqual(b.models['denoiser'].other_complex.item(),2j)
   la=update(a);lb=update(b);self.assertTrue(torch.equal(la,lb))
   for pa,pb in zip(a.master_params,b.master_params):self.assertTrue(torch.equal(pa,pb))
 # 契约：加载前先校验检查点，文件被篡改要在触碰模型之前失败。
 def test_live_load_tamper_refused_before_model(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();p=Path(td)/'ckpts/denoiser_step0000003.pt';p.write_bytes(b'x');b=tiny(td)
   with self.assertRaises(ValueError):b.load(td,3)
 # 契约：训练拓扑（如 ema_rate）与检查点不一致时拒绝加载，必须报 'configured'。
 def test_live_load_topology_mismatch(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();b=tiny(td);b.ema_rate=[.9]
   with self.assertRaisesRegex(ValueError,'configured'):b.load(td,3)
 # 契约：旧版无 marker 但文件齐全的检查点允许加载，但必须发出 RuntimeWarning（且不补写 marker）。
 def test_live_explicit_legacy_complete_allowed_with_warning(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();marker=Path(td)/'ckpts'/cc.marker_name(3);backup=marker.read_bytes();marker.unlink() # temp fixture only
   b=tiny(td)
   with self.assertWarns(RuntimeWarning):b.load(td,3)
   self.assertFalse(marker.exists())
 # 契约：旧版且不完整的检查点必须拒绝（不允许部分恢复）。
 def test_live_legacy_incomplete_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/'ckpts').mkdir();torch.save({},Path(td)/'ckpts/misc_step0000003.pt')
   with self.assertRaises(ValueError):tiny(td).load(td,3)
 # 契约：finetune_from 载入源权重时保留复数 buffer、LoRA 权重精确不变，且 master_params 为 fp32。
 def test_live_finetune_preserves_complex_and_exact_fp32(self):
  with tempfile.TemporaryDirectory() as td:
   t=tiny(td);state=t.models['denoiser'].state_dict();p=Path(td)/'source.pt';torch.save(state,p);before=state['proj.lora_down.weight'].clone();t.finetune_from({'denoiser':str(p)})
   self.assertEqual(t.models['denoiser'].rope_phases.item(),1j);self.assertTrue(torch.equal(before,t.models['denoiser'].proj.lora_down.weight));self.assertEqual(t.master_params[0].dtype,torch.float32)
 # 契约：non_blocking=False 同样是同步落盘，步骤号照常记录。
 def test_nonblocking_false_is_also_synchronous(self):
  with tempfile.TemporaryDirectory() as td:
   m=cc.atomic_save_group(td,3,payload(),non_blocking=False);self.assertFalse(m['requested_non_blocking']);self.assertEqual(cc.list_complete_steps(td),[3])
 # 契约：world_size>1 时 rank0 的 RNG 只对 rank0 生效，恢复要告警（其余 rank 不允许共用）。
 def test_rank0_ddp_rng_scope_warns(self):
  s=cc.capture_rng_state(world_size=2)
  with self.assertWarns(RuntimeWarning):self.assertTrue(cc.restore_rng_state(s,rank=0,world_size=2))
 # 契约：optimizer 动量与 step 一律对齐到 fp32（避免 bfloat16 累积误差）。
 def test_model_base_no_adapter_rounding_and_moment_alignment(self):
  m=Model();opt=torch.optim.AdamW(m.proj.lora_down.parameters());p=m.proj.lora_down.weight
  opt.state[p]={'exp_avg':torch.ones_like(p,dtype=torch.bfloat16),'step':torch.tensor(1.)}
  cc.align_optimizer_moments(opt);self.assertEqual(opt.state[p]['exp_avg'].dtype,torch.float32);self.assertEqual(opt.state[p]['step'].dtype,torch.float32)
 # 契约：misc 检查点里的 step 与请求的 step 不一致时拒绝加载（报 'misc checkpoint step'）。
 def test_live_misc_wrong_step_refused(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();marker=Path(td)/'ckpts'/cc.marker_name(3)
   # Mutate only temporary legacy fixture; production ckpts untouched.
    # 中文：先断言待改的 marker 确实位于临时目录内，确保不会误改生产检查点。
   assert marker.resolve().parent==Path(td).resolve()/'ckpts';marker.unlink()
   p=Path(td)/'ckpts/misc_step0000003.pt';m=torch.load(p,weights_only=False);m['step']=5;torch.save(m,p)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    with self.assertRaisesRegex(ValueError,'misc checkpoint step'):tiny(td).load(td,3)
 # 契约：抽取真实 run() 里的“最终保存”分支——非 interval 的末尾保存要生效，重复调用不覆盖，
 # 恢复后已提交的 finalstep 只校验不重写，非 master 不保存。
 def test_actual_run_final_save_noninterval_and_no_duplicate(self):
  run=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  branch=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and 'last_committed_step' in ast.unparse(n))
  env={'os':os,'marker_name':cc.marker_name,'validate_checkpoint':cc.validate_checkpoint}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='final_save',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[branch],decorator_list=[])],type_ignores=[])),'live-final-save','exec'),env)
  with tempfile.TemporaryDirectory() as td:
   t=tiny(td);env['final_save'](t);self.assertEqual(cc.list_complete_steps(Path(td)/'ckpts'),[3]);env['final_save'](t)
   # A resumed run at already committed finalstep verifies but does not overwrite.
   del t._last_committed_step;env['final_save'](t)
   t.step=4;t.is_master=False;env['final_save'](t);self.assertEqual(cc.list_complete_steps(Path(td)/'ckpts'),[3])
 # 契约：抽取真实 run() 的快照分支——snapshot 抛错必须向上传播（不得吞掉）。
 def test_actual_snapshot_error_propagates(self):
  run=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  branch=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and getattr(n.test.func,'id',None)=='snapshot_allowed')
  env={'snapshot_allowed':cc.snapshot_allowed}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='final_branch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[branch],decorator_list=[])],type_ignores=[])),'live-final-snapshot','exec'),env)
  s=types.SimpleNamespace(dataset=object(),snapshot_batch_size=2,snapshot=lambda **kw:(_ for _ in ()).throw(RuntimeError('sample failed')))
  with self.assertRaisesRegex(RuntimeError,'sample failed'):env['final_branch'](s)
 # 契约：抽取真实 run() 的快照分支——dataset.skip_snapshot 为真时跳过，为假时恰好调用一次。
 def test_actual_final_run_branch_skip_and_enabled(self):
  run=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='run');branch=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and getattr(n.test.func,'id',None)=='snapshot_allowed')
  testns={'snapshot_allowed':cc.snapshot_allowed};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='final_branch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[branch],decorator_list=[])],type_ignores=[])),'live-final-snapshot','exec'),testns)
  for skip,count in [(True,0),(False,1)]:
   calls=[];s=types.SimpleNamespace(dataset=types.SimpleNamespace(skip_snapshot=skip),snapshot_batch_size=2,snapshot=lambda **kw:calls.append(kw));testns['final_branch'](s);self.assertEqual(len(calls),count)

if __name__=='__main__':unittest.main(verbosity=2)
