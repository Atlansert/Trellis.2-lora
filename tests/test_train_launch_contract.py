"""训练启动契约（trellis2.utils.train_launch_contract 与 train.py 的启动块）的 CPU 单元测试。

- 覆盖点：检查点选择（fresh/none/显式 step/latest）与历史保护、latest 只认已提交的完整步骤
  （忽略未提交的 misc）、选中步骤被篡改或 marker 损坏时 fail-closed、legacy 数字步骤需完整一组并告警、
  外部 load_dir 恢复不得与已有 output_dir 混合、启动记录（config/command/run_attempts）保留历史且文件名唯一、
  重试语义（0 次=1 次尝试、none 失败不得自动从 0 重启、重试改从本地已提交 latest 恢复、
  失败写入 run_attempts 且状态 failed）、子进程失败退出码非 0（CUDA_VISIBLE_DEVICES 置空，纯 CPU）、
  以及用 AST 抽取 train.py 真实启动块与 find_ckpt，验证“守卫先于历史写入”“不覆盖旧 config/command”。
- 运行：python -m pytest tests/test_train_launch_contract.py（纯 CPU；会 spawn 一个 python 子进程）。
- 依赖：torch、trellis2.utils.train_launch_contract、trellis2.utils.checkpoint_contract，并按相对路径读 train.py。
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from trellis2.utils.train_launch_contract import (resolve_checkpoint,expected_files,preserve_launch_records,run_attempts,load_pretrained_checked)


# 测试用训练配置：output_dir/load_dir/ckpt 选择与 auto_retry 次数，外加最小的 models/trainer 字段。
def cfg(root, choice='latest', load=None, retry=3):
    return dict(output_dir=str(root),load_dir=str(load or root),ckpt=choice,auto_retry=retry,
                models={'denoiser':{}},trainer={'args':{'ema_rate':0.9999}})


# 在 <output_dir>/ckpts 下补齐该 step 应有的一整组文件（内容为占位字节）。
def group(c,step):
    p=Path(c['output_dir'])/'ckpts';p.mkdir(exist_ok=True)
    for name in expected_files(c,step):(p/name).write_bytes(b'tiny')
    return p


class LaunchTests(unittest.TestCase):
    # 契约：全新目录 + latest/none 时不加载任何检查点，且解析过程不得创建 output_dir。
    def test_empty_fresh_latest_explicit_none(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(Path(td)/'new');resolve_checkpoint(c,lambda p:[]);self.assertIsNone(c['load_ckpt']);self.assertFalse(Path(c['output_dir']).exists())
            resolve_checkpoint(cfg(Path(td)/'new','none'));self.assertFalse(Path(c['output_dir']).exists())
    # 契约：目录里只有文件、没有已提交 marker 时，latest 不得猜出步骤号（要求显式数字）。
    def test_latest_requires_marker_not_misc(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);group(c,5)
            with self.assertRaisesRegex(ValueError,'explicit numeric'):resolve_checkpoint(c,lambda p:[])
    # 契约：output_dir 里已有 config.json（说明是续训场景）时，none/latest 必须报错，
 # 且不修改既有 config 字节。
    def test_config_history_blocks_none_and_empty_latest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_bytes(b'original')
            for choice in ['none','latest']:
                with self.assertRaises(ValueError):resolve_checkpoint(cfg(td,choice),lambda p:[])
            self.assertEqual(p.read_bytes(),b'original')
    # 契约：latest 应选“已提交且通过校验”的 7，而不是数字更大的未提交 900。
    def test_latest_selects_complete_not_highest_misc(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);group(c,7);group(c,900);calls=[]
            resolve_checkpoint(c,lambda p:[7],lambda p,s:calls.append(s))
            self.assertEqual(c['load_ckpt'],7);self.assertEqual(calls,[7])
    # 契约：选中的完整检查点若哈希校验失败，必须报错（不得退回更旧的步骤）。
    def test_selected_complete_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);group(c,7)
            def bad(p,s):raise ValueError('hash mismatch')
            with self.assertRaisesRegex(ValueError,'hash'):resolve_checkpoint(c,lambda p:[7],bad)
    # 契约：显式数字步骤走 legacy 兼容路径（告警 LEGACY）；若该 step 的文件组不完整则报 incomplete。
    def test_legacy_numeric_warns_requires_full_group(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'7');p=group(c,7)
            with self.assertWarnsRegex(RuntimeWarning,'LEGACY'):resolve_checkpoint(c)
            self.assertEqual(c['load_ckpt'],7)
            (p/'denoiser_ema0.9999_step0000007.pt').write_bytes(b'')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with self.assertRaisesRegex(ValueError,'incomplete'):resolve_checkpoint(c)
    # 契约：显式数字步骤的 marker 损坏时直接失败，不得降级成 legacy 兼容。
    def test_numeric_marker_tamper_not_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'7');p=group(c,7);(p/'complete_step0000007.json').write_text('{}')
            def invalid(p,s):raise ValueError('invalid marker')
            with self.assertRaisesRegex(ValueError,'invalid marker'):resolve_checkpoint(c,validate=invalid)
    # 契约：从另一个目录恢复并输出到新目录时告警，且不创建新 output_dir。
    def test_distinct_load_resume_to_new_output(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'old';source.mkdir();group(cfg(source),7)
            c=cfg(Path(td)/'new','7',source)
            with self.assertWarns(RuntimeWarning):resolve_checkpoint(c)
            self.assertEqual(c['load_ckpt'],7);self.assertFalse(Path(c['output_dir']).exists())
    # 契约：外部恢复不允许与已存在的 output_dir 混用（报 'cannot be mixed'），目录内容保持不变。
    def test_external_resume_cannot_mix_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'old';source.mkdir();group(cfg(source),7)
            output=Path(td)/'existing';output.mkdir();f=output/'config.json';f.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError,'cannot be mixed'):resolve_checkpoint(cfg(output,'7',source))
            self.assertEqual(f.read_bytes(),b'original');self.assertEqual(list(output.iterdir()),[f])
    # 契约：外部恢复目录不存在时（latest 却为空）必须报错，不能悄悄当成全新训练。
    def test_external_empty_latest_not_fresh_output(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):resolve_checkpoint(cfg(Path(td)/'new','latest',Path(td)/'absent'),lambda p:[])
    # 契约：启动记录每次写到新的唯一路径（不覆盖），旧的 config/command 原样保留，
 # 记录里含 resolved_checkpoint 与 64 位 sha256 溯源。
    def test_preserves_history_and_unique_attempt_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'config.json').write_bytes(b'oldconfig');(root/'command.txt').write_bytes(b'oldcommand');c=cfg(root,'7');c['load_ckpt']=7
            paths=[]
            for i in range(2):paths.append(preserve_launch_records(c,{'new':1},'resume command',__file__)[0])
            self.assertNotEqual(*paths);self.assertEqual((root/'config.json').read_bytes(),b'oldconfig');self.assertEqual((root/'command.txt').read_bytes(),b'oldcommand')
            rec=json.loads(paths[0].read_text());self.assertEqual(rec['resolved_checkpoint'],7);self.assertEqual(len(rec['source_sha256']),64)
    # 契约：首次为 none（无检查点）且失败时，不允许自动重试（否则会从 0 重来），只尝试一次。
    def test_first_none_failure_never_restarts_zero(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'none');c['load_ckpt']=None;seen=[]
            def fail(c):seen.append(c['load_ckpt']);raise ValueError('training failed')
            with self.assertRaisesRegex(RuntimeError,'safe retry unavailable'):run_attempts(c,fail,complete_steps=lambda p:[])
            self.assertEqual(seen,[None])
    # 契约：从外部目录初始恢复、但本次没提交任何输出时，重试会“重放”同一状态，必须拒绝（报 'replay'）。
    def test_external_initial_resume_no_output_commit_refuses_replay(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'7','elsewhere');c['load_ckpt']=7;seen=[]
            def fail(c):seen.append(c['load_ckpt']);raise ValueError('failure')
            with self.assertRaisesRegex(RuntimeError,'replay'):run_attempts(c,fail,complete_steps=lambda p:[])
            self.assertEqual(seen,[7])
    # 契约：首次失败后若本地已产生已提交的 step，重试改为从该 latest 恢复（ckpt 也被改成 latest）。
    def test_retry_changes_none_to_local_committed_latest(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'none');c['load_ckpt']=None;seen=[]
            def run(c):
                seen.append((c['load_dir'],c['load_ckpt']))
                if len(seen)==1:group(c,5);raise ValueError('failure')
            run_attempts(c,run,complete_steps=lambda p:[5],validate=lambda p,s:None)
            self.assertEqual([x[1] for x in seen],[None,5]);self.assertEqual(c['ckpt'],'latest')
    # 契约：重试次数用尽时恰好尝试 3 次、打印 2 次 [Retry]，并把记录状态置为 failed。
    def test_maxattempts_final_failure_no_extra_retry_message(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,retry=3);c['load_ckpt']=5;group(c,5);seen=[];text=io.StringIO();record={'attempts':[]}
            def fail(c):seen.append(c['load_ckpt']);raise ValueError('failed')
            with contextlib.redirect_stdout(text),self.assertRaisesRegex(RuntimeError,'exhausted 3'):run_attempts(c,fail,record=record,complete_steps=lambda p:[5],validate=lambda p,s:None)
            self.assertEqual(seen,[5,5,5]);self.assertEqual(text.getvalue().count('[Retry]'),2);self.assertEqual(record['status'],'failed')
    # 契约：auto_retry=0 表示只尝试一次（不重试）。
    def test_zero_retry_means_one_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,retry=0);c['load_ckpt']=None;seen=[]
            def fail(c):seen.append(1);raise ValueError()
            with self.assertRaisesRegex(RuntimeError,'exhausted 1'):run_attempts(c,fail)
            self.assertEqual(seen,[1])
    # 契约：在独立子进程里跑失败路径，进程退出码必须非 0，且 stderr 含 'exhausted 1'
 # （CUDA_VISIBLE_DEVICES 置空以保证纯 CPU 环境也能跑）。
    def test_failure_subprocess_nonzero_cpu_only(self):
        code="from trellis2.utils.train_launch_contract import run_attempts\nc={'output_dir':'unused','load_dir':'unused','ckpt':'none','load_ckpt':None,'auto_retry':0}\ndef fail(c):raise ValueError('stub')\nrun_attempts(c,fail)"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertNotEqual(result.returncode,0);self.assertIn('exhausted 1',result.stderr)
    # 契约：失败必须落盘到 run_attempts 记录（status=failed、含 traceback 与 finished_utc）。
    def test_attempt_record_failure_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'none',retry=0);c['load_ckpt']=None;p,r=preserve_launch_records(c,{},'command',__file__)
            def fail(c):raise ValueError('error')
            with self.assertRaises(RuntimeError):run_attempts(c,fail,p,r)
            data=json.loads(p.read_text());self.assertEqual(data['status'],'failed');self.assertIn('traceback',data['attempts'][0]);self.assertTrue(data['finished_utc'])
    # 契约：用真实 atomic_save_group 写一组完整检查点，latest 选中 7；旁边未提交的 900 被忽略。
    def test_real_atomic_latest_ignores_uncommitted_misc(self):
        import torch
        from trellis2.utils.checkpoint_contract import atomic_save_group
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);p=Path(td)/'ckpts'
            payloads={name:{'tiny':torch.ones(1)} for name in expected_files(c,7)}
            atomic_save_group(p,7,payloads)
            (p/'misc_step0000900.pt').write_bytes(b'incomplete')
            resolve_checkpoint(c)
            self.assertEqual(c['load_ckpt'],7)
    # 契约：真实检查点被翻转 1 bit，或 marker JSON 损坏时，解析都必须 fail-closed（抛 ValueError）。
    def test_real_atomic_tamper_and_badmarker_failclosed(self):
        import torch
        from trellis2.utils.checkpoint_contract import atomic_save_group
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);p=Path(td)/'ckpts'
            atomic_save_group(p,7,{name:{'tiny':torch.ones(1)} for name in expected_files(c,7)})
            target=p/'denoiser_step0000007.pt';data=target.read_bytes();target.write_bytes(bytes([data[0]^1])+data[1:])
            with self.assertRaises(ValueError):resolve_checkpoint(c)
            target.write_bytes(data)
            (p/'complete_step0000900.json').write_text('{broken')
            with self.assertRaises(ValueError):resolve_checkpoint(c)
    # 契约（源码级）：train.py 必须“先 find_ckpt 守卫、后写启动记录”，且不得再用
 # 直接 open(..., 'w') 的方式覆盖 config.json / command.txt。
    def test_fresh_guard_ast_precedes_history_write(self):
        source=Path('train.py').read_text()
        self.assertLess(source.index('cfg = find_ckpt(cfg)'),source.index('record_path, record = preserve_launch_records'))
        self.assertNotIn("with open(os.path.join(cfg.output_dir, 'config.json'), 'w')",source)
        self.assertNotIn("with open(os.path.join(cfg.output_dir, 'command.txt'), 'w')",source)

    # 用 AST 抽出 train.py 的 __main__ 启动块，在一个受控命名空间里 exec（main 换成桩函数），
 # 从而在 CPU 上验证真实的参数解析与启动/重试流程。
    def execute_launch_block(self,arguments,runner):
        import argparse,types
        from easydict import EasyDict
        source=Path('train.py').read_text();tree=ast.parse(source)
        block=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and getattr(n.test.left,'id',None)=='__name__')
        namespace={'__name__':'__main__','__file__':str(Path('train.py').resolve()),
            'sys':sys,
            'argparse':argparse,'json':json,'edict':EasyDict,
            'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda:1)),
            'find_ckpt':resolve_checkpoint,'preserve_launch_records':preserve_launch_records,
            'run_attempts':run_attempts,'main':runner}
        from unittest.mock import patch
        with patch.object(sys,'argv',['train.py']+arguments):
            exec(compile(ast.Module(body=[block],type_ignores=[]),'train.py','exec'),namespace)
    # 契约：真实启动块在训练最终失败时向上抛 RuntimeError，并写出 1 份 run_attempts 记录（failed）。
    def test_actual_ast_launch_final_failure_propagates_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);conf=root/'input.json';conf.write_text(json.dumps({'models':{'denoiser':{}},'trainer':{'args':{}}}));out=root/'fresh';calls=[]
            def stub(rank,cfg):calls.append(cfg.load_ckpt);raise ValueError('CPU stub failure')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'exhausted 1'):
                self.execute_launch_block(['--config',str(conf),'--output_dir',str(out),'--ckpt','none','--num_gpus','1','--auto_retry','0'],stub)
            self.assertEqual(calls,[None]);logs=list((out/'run_attempts').glob('*.json'));self.assertEqual(len(logs),1)
            self.assertEqual(json.loads(logs[0].read_text())['status'],'failed')
    # 契约：真实启动块遇到已存在的 output_dir 历史时，在调用训练 main 之前就报 'overwrite'，
 # 且目录内文件的字节与 mtime 全部不变。
    def test_actual_ast_launch_guard_preserves_old_history_no_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);conf=root/'input.json';conf.write_text(json.dumps({'models':{'denoiser':{}},'trainer':{'args':{}}}));out=root/'existing';out.mkdir();(out/'config.json').write_bytes(b'original config');(out/'command.txt').write_bytes(b'original command')
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()};calls=[]
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'overwrite'):
                self.execute_launch_block(['--config',str(conf),'--output_dir',str(out),'--ckpt','none','--num_gpus','1'],lambda r,c:calls.append(c))
            self.assertEqual(calls,[]);self.assertEqual({p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()},before)

    # 契约：train.py 的 find_ckpt 只做委托（转调 resolve_checkpoint），源码里不再出现 misc_*.pt 逻辑。
    def test_train_ast_findckpt_delegates(self):
        source=Path('train.py').read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='find_ckpt');space={'resolve_checkpoint':lambda c:dict(c,passed=True)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'train.py','exec'),space)
        self.assertTrue(space['find_ckpt']({})['passed']);self.assertNotIn('misc_*.pt',source)
    # 契约：预训练权重按严格校验加载（rope_phases 仍由代码重算而非常量），键缺失时报 RuntimeError。
    def test_pretrained_rope_preserved_and_bad_keys_fail(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.register_buffer('rope_phases',torch.tensor([1j]))
        model=Tiny();load_pretrained_checked(model,{'weight':torch.ones(1),'rope_phases':torch.zeros(1)});self.assertEqual(model.rope_phases.item(),1j)
        with self.assertRaises(RuntimeError):load_pretrained_checked(model,{'other':torch.ones(1)})

if __name__=='__main__':unittest.main(verbosity=2)
