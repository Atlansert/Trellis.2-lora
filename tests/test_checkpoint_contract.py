import ast, contextlib, copy, io, json, os, random, tempfile, types, unittest, warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from trellis2.modules.lora import LoRALinear
from trellis2.utils import checkpoint_contract as cc

# Exercise exact live BasicTrainer methods without importing CUDA trainer setup.
source=Path('trellis2/trainers/basic.py').read_text()
module=ast.parse(source);klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='BasicTrainer')
names=['load','save','finetune_from','_master_params_to_state_dicts','_state_dicts_to_master_params']
methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in names]
ns=dict(vars(cc));ns.update(torch=torch,os=os,warnings=warnings,read_file_dist=lambda p:p)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='TinyBasic',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),'live-basic-methods','exec'),ns)
TinyBasic=ns['TinyBasic']

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

def tiny(td,model=None):
 t=TinyBasic();t.models={'denoiser':model or Model()};t.master_params=[p for p in t.models['denoiser'].parameters() if p.requires_grad];t.model_params=t.master_params
 t.ema_rate=[.9999];t.ema_params=[copy.deepcopy(t.master_params)];t.optimizer=torch.optim.AdamW(t.master_params,lr=.01)
 t.is_master=True;t.rank=0;t.world_size=1;t.device=torch.device('cpu');t.mix_precision_mode='amp';t.mix_precision_dtype=torch.bfloat16;t.output_dir=td;t.step=3;t.data_sampler=Sampler()
 t.lr_scheduler_config=None;t.elastic_controller_config=None;t.grad_clip=None
 return t

def update(t):
 t.optimizer.zero_grad();x=torch.tensor([[.4,.2,-.3]],dtype=torch.bfloat16)
 with torch.autocast('cpu',dtype=torch.bfloat16):loss=t.models['denoiser'](x).float().square().sum()
 loss.backward();t.optimizer.step();return loss.detach()

def payload(step=3):return {f'denoiser_step{step:07d}.pt':{'x':torch.tensor([1.25])},f'misc_step{step:07d}.pt':{'step':step}}

class CheckpointTests(unittest.TestCase):
 def test_atomic_complete_files_hashes_and_sync(self):
  with tempfile.TemporaryDirectory() as td:
   m=cc.atomic_save_group(td,3,payload(),non_blocking=True)
   self.assertTrue(m['synchronous']);self.assertTrue(m['requested_non_blocking']);self.assertEqual(cc.list_complete_steps(td),[3]);self.assertEqual(cc.validate_checkpoint(td,3),m)
 def test_no_marker_for_save_failure(self):
  with tempfile.TemporaryDirectory() as td:
   with patch.object(cc.torch,'save',side_effect=OSError('disk full')):
    with self.assertRaises(OSError):cc.atomic_save_group(td,3,payload())
   self.assertEqual(cc.list_complete_steps(td),[]);self.assertFalse((Path(td)/cc.marker_name(3)).exists())
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
 def test_complete_overwrite_refused_unchanged(self):
  with tempfile.TemporaryDirectory() as td:
   cc.atomic_save_group(td,3,payload());before=(Path(td)/cc.marker_name(3)).read_bytes()
   with self.assertRaises(FileExistsError):cc.atomic_save_group(td,3,payload())
   self.assertEqual(before,(Path(td)/cc.marker_name(3)).read_bytes())
 def test_partial_target_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'denoiser_step0000003.pt';p.write_bytes(b'old')
   with self.assertRaises(FileExistsError):cc.atomic_save_group(td,3,payload())
   self.assertEqual(p.read_bytes(),b'old')
 def test_bad_marker_raises_not_silent_latest(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/cc.marker_name(3)).write_text('{}')
   with self.assertRaises(ValueError):cc.list_complete_steps(td)
 def test_empty_misconly_unsafe_wrong_step_refused(self):
  with tempfile.TemporaryDirectory() as td:
   for p in [{},{'misc_step0000003.pt':{}},{'../x_step0000003.pt':{},'misc_step0000003.pt':{}},{'x_step0000004.pt':{},'misc_step0000003.pt':{}}]:
    with self.assertRaises(ValueError):cc.atomic_save_group(td,3,p)
 def test_tamper_same_size_hash_failure(self):
  with tempfile.TemporaryDirectory() as td:
   cc.atomic_save_group(td,3,payload());p=Path(td)/'denoiser_step0000003.pt';b=bytearray(p.read_bytes());b[-1]^=1;p.write_bytes(b)
   self.assertEqual(cc.list_complete_steps(td,False),[3])
   with self.assertRaises(ValueError):cc.validate_checkpoint(td,3,True)
 def test_missing_size_mismatch(self):
  with tempfile.TemporaryDirectory() as td:
   cc.atomic_save_group(td,3,payload());(Path(td)/'misc_step0000003.pt').write_bytes(b'x')
   with self.assertRaises(ValueError):cc.list_complete_steps(td)
 def test_legacy_files_not_complete(self):
  with tempfile.TemporaryDirectory() as td:
   torch.save({},Path(td)/'misc_step0000003.pt');self.assertEqual(cc.list_complete_steps(td),[])
 def test_snapshot_owns_cpu_storage(self):
  t=torch.ones(2);s=cc.cpu_snapshot({'x':t});t.fill_(8);self.assertTrue(s['x'].eq(1).all())
 def test_rng_roundtrip_python_numpy_torch(self):
  random.seed(17);np.random.seed(17);torch.manual_seed(17);s=cc.capture_rng_state();a=(random.random(),np.random.rand(),torch.rand(2));cc.restore_rng_state(s);b=(random.random(),np.random.rand(),torch.rand(2));self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))
 def test_other_rank_does_not_receive_rank0_rng(self):
  s=cc.capture_rng_state();torch.manual_seed(23);before=torch.get_rng_state().clone()
  with self.assertWarns(RuntimeWarning):self.assertFalse(cc.restore_rng_state(s,rank=1,world_size=2))
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
 def test_legacy_rng_warn(self):
  with self.assertWarns(RuntimeWarning):self.assertFalse(cc.restore_rng_state(None))
 def test_cast_exact_adapters_identity_and_all_complex(self):
  m=Model();p=m.proj.lora_down.weight;value=p.detach().clone();complex0=m.other_complex
  cc.cast_model_preserving_lora_and_complex(m,torch.bfloat16)
  self.assertIs(m.proj.lora_down.weight,p);self.assertTrue(torch.equal(p,value));self.assertEqual(p.dtype,torch.float32);self.assertEqual(m.proj.weight.dtype,torch.bfloat16);self.assertIs(m.other_complex,complex0);self.assertEqual(m.rope_phases.item(),1j)
 def test_nonlora_model_not_forced_fp32(self):
  m=torch.nn.Linear(3,2);cc.cast_model_preserving_lora_and_complex(m,torch.bfloat16);self.assertEqual(m.weight.dtype,torch.bfloat16)
 def test_live_basic_save_load_next_step_equivalent(self):
  with tempfile.TemporaryDirectory() as td:
   torch.manual_seed(4);a=tiny(td);update(a);a.save(non_blocking=True);b=tiny(td);refs=list(b.master_params);b.load(td,3)
   self.assertEqual(b.step,3);self.assertEqual(b.data_sampler.state,{'position':7})
   for pa,pb,ref in zip(a.master_params,b.master_params,refs):
    self.assertIs(pb,ref);self.assertEqual(pb.dtype,torch.float32);self.assertTrue(torch.equal(pa,pb));self.assertEqual(b.optimizer.state[pb]['exp_avg'].dtype,torch.float32)
   self.assertEqual(b.models['denoiser'].rope_phases.item(),1j);self.assertEqual(b.models['denoiser'].other_complex.item(),2j)
   la=update(a);lb=update(b);self.assertTrue(torch.equal(la,lb))
   for pa,pb in zip(a.master_params,b.master_params):self.assertTrue(torch.equal(pa,pb))
 def test_live_load_tamper_refused_before_model(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();p=Path(td)/'ckpts/denoiser_step0000003.pt';p.write_bytes(b'x');b=tiny(td)
   with self.assertRaises(ValueError):b.load(td,3)
 def test_live_load_topology_mismatch(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();b=tiny(td);b.ema_rate=[.9]
   with self.assertRaisesRegex(ValueError,'configured'):b.load(td,3)
 def test_live_explicit_legacy_complete_allowed_with_warning(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();marker=Path(td)/'ckpts'/cc.marker_name(3);backup=marker.read_bytes();marker.unlink() # temp fixture only
   b=tiny(td)
   with self.assertWarns(RuntimeWarning):b.load(td,3)
   self.assertFalse(marker.exists())
 def test_live_legacy_incomplete_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   (Path(td)/'ckpts').mkdir();torch.save({},Path(td)/'ckpts/misc_step0000003.pt')
   with self.assertRaises(ValueError):tiny(td).load(td,3)
 def test_live_finetune_preserves_complex_and_exact_fp32(self):
  with tempfile.TemporaryDirectory() as td:
   t=tiny(td);state=t.models['denoiser'].state_dict();p=Path(td)/'source.pt';torch.save(state,p);before=state['proj.lora_down.weight'].clone();t.finetune_from({'denoiser':str(p)})
   self.assertEqual(t.models['denoiser'].rope_phases.item(),1j);self.assertTrue(torch.equal(before,t.models['denoiser'].proj.lora_down.weight));self.assertEqual(t.master_params[0].dtype,torch.float32)
 def test_nonblocking_false_is_also_synchronous(self):
  with tempfile.TemporaryDirectory() as td:
   m=cc.atomic_save_group(td,3,payload(),non_blocking=False);self.assertFalse(m['requested_non_blocking']);self.assertEqual(cc.list_complete_steps(td),[3])
 def test_rank0_ddp_rng_scope_warns(self):
  s=cc.capture_rng_state(world_size=2)
  with self.assertWarns(RuntimeWarning):self.assertTrue(cc.restore_rng_state(s,rank=0,world_size=2))
 def test_model_base_no_adapter_rounding_and_moment_alignment(self):
  m=Model();opt=torch.optim.AdamW(m.proj.lora_down.parameters());p=m.proj.lora_down.weight
  opt.state[p]={'exp_avg':torch.ones_like(p,dtype=torch.bfloat16),'step':torch.tensor(1.)}
  cc.align_optimizer_moments(opt);self.assertEqual(opt.state[p]['exp_avg'].dtype,torch.float32);self.assertEqual(opt.state[p]['step'].dtype,torch.float32)
 def test_live_misc_wrong_step_refused(self):
  with tempfile.TemporaryDirectory() as td:
   a=tiny(td);a.save();marker=Path(td)/'ckpts'/cc.marker_name(3)
   # Mutate only temporary legacy fixture; production ckpts untouched.
   assert marker.resolve().parent==Path(td).resolve()/'ckpts';marker.unlink()
   p=Path(td)/'ckpts/misc_step0000003.pt';m=torch.load(p,weights_only=False);m['step']=5;torch.save(m,p)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    with self.assertRaisesRegex(ValueError,'misc checkpoint step'):tiny(td).load(td,3)
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
 def test_actual_snapshot_error_propagates(self):
  run=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  branch=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and getattr(n.test.func,'id',None)=='snapshot_allowed')
  env={'snapshot_allowed':cc.snapshot_allowed}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='final_branch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[branch],decorator_list=[])],type_ignores=[])),'live-final-snapshot','exec'),env)
  s=types.SimpleNamespace(dataset=object(),snapshot_batch_size=2,snapshot=lambda **kw:(_ for _ in ()).throw(RuntimeError('sample failed')))
  with self.assertRaisesRegex(RuntimeError,'sample failed'):env['final_branch'](s)
 def test_actual_final_run_branch_skip_and_enabled(self):
  run=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='run');branch=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and getattr(n.test.func,'id',None)=='snapshot_allowed')
  testns={'snapshot_allowed':cc.snapshot_allowed};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='final_branch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[branch],decorator_list=[])],type_ignores=[])),'live-final-snapshot','exec'),testns)
  for skip,count in [(True,0),(False,1)]:
   calls=[];s=types.SimpleNamespace(dataset=types.SimpleNamespace(skip_snapshot=skip),snapshot_batch_size=2,snapshot=lambda **kw:calls.append(kw));testns['final_branch'](s);self.assertEqual(len(calls),count)

if __name__=='__main__':unittest.main(verbosity=2)
