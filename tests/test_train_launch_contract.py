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


def cfg(root, choice='latest', load=None, retry=3):
    return dict(output_dir=str(root),load_dir=str(load or root),ckpt=choice,auto_retry=retry,
                models={'denoiser':{}},trainer={'args':{'ema_rate':0.9999}})


def group(c,step):
    p=Path(c['output_dir'])/'ckpts';p.mkdir(exist_ok=True)
    for name in expected_files(c,step):(p/name).write_bytes(b'tiny')
    return p


class LaunchTests(unittest.TestCase):
    def test_empty_fresh_latest_explicit_none(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(Path(td)/'new');resolve_checkpoint(c,lambda p:[]);self.assertIsNone(c['load_ckpt']);self.assertFalse(Path(c['output_dir']).exists())
            resolve_checkpoint(cfg(Path(td)/'new','none'));self.assertFalse(Path(c['output_dir']).exists())
    def test_latest_requires_marker_not_misc(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);group(c,5)
            with self.assertRaisesRegex(ValueError,'explicit numeric'):resolve_checkpoint(c,lambda p:[])
    def test_config_history_blocks_none_and_empty_latest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_bytes(b'original')
            for choice in ['none','latest']:
                with self.assertRaises(ValueError):resolve_checkpoint(cfg(td,choice),lambda p:[])
            self.assertEqual(p.read_bytes(),b'original')
    def test_latest_selects_complete_not_highest_misc(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);group(c,7);group(c,900);calls=[]
            resolve_checkpoint(c,lambda p:[7],lambda p,s:calls.append(s))
            self.assertEqual(c['load_ckpt'],7);self.assertEqual(calls,[7])
    def test_selected_complete_tamper_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td);group(c,7)
            def bad(p,s):raise ValueError('hash mismatch')
            with self.assertRaisesRegex(ValueError,'hash'):resolve_checkpoint(c,lambda p:[7],bad)
    def test_legacy_numeric_warns_requires_full_group(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'7');p=group(c,7)
            with self.assertWarnsRegex(RuntimeWarning,'LEGACY'):resolve_checkpoint(c)
            self.assertEqual(c['load_ckpt'],7)
            (p/'denoiser_ema0.9999_step0000007.pt').write_bytes(b'')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with self.assertRaisesRegex(ValueError,'incomplete'):resolve_checkpoint(c)
    def test_numeric_marker_tamper_not_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'7');p=group(c,7);(p/'complete_step0000007.json').write_text('{}')
            def invalid(p,s):raise ValueError('invalid marker')
            with self.assertRaisesRegex(ValueError,'invalid marker'):resolve_checkpoint(c,validate=invalid)
    def test_distinct_load_resume_to_new_output(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'old';source.mkdir();group(cfg(source),7)
            c=cfg(Path(td)/'new','7',source)
            with self.assertWarns(RuntimeWarning):resolve_checkpoint(c)
            self.assertEqual(c['load_ckpt'],7);self.assertFalse(Path(c['output_dir']).exists())
    def test_external_resume_cannot_mix_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'old';source.mkdir();group(cfg(source),7)
            output=Path(td)/'existing';output.mkdir();f=output/'config.json';f.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError,'cannot be mixed'):resolve_checkpoint(cfg(output,'7',source))
            self.assertEqual(f.read_bytes(),b'original');self.assertEqual(list(output.iterdir()),[f])
    def test_external_empty_latest_not_fresh_output(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):resolve_checkpoint(cfg(Path(td)/'new','latest',Path(td)/'absent'),lambda p:[])
    def test_preserves_history_and_unique_attempt_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'config.json').write_bytes(b'oldconfig');(root/'command.txt').write_bytes(b'oldcommand');c=cfg(root,'7');c['load_ckpt']=7
            paths=[]
            for i in range(2):paths.append(preserve_launch_records(c,{'new':1},'resume command',__file__)[0])
            self.assertNotEqual(*paths);self.assertEqual((root/'config.json').read_bytes(),b'oldconfig');self.assertEqual((root/'command.txt').read_bytes(),b'oldcommand')
            rec=json.loads(paths[0].read_text());self.assertEqual(rec['resolved_checkpoint'],7);self.assertEqual(len(rec['source_sha256']),64)
    def test_first_none_failure_never_restarts_zero(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'none');c['load_ckpt']=None;seen=[]
            def fail(c):seen.append(c['load_ckpt']);raise ValueError('training failed')
            with self.assertRaisesRegex(RuntimeError,'safe retry unavailable'):run_attempts(c,fail,complete_steps=lambda p:[])
            self.assertEqual(seen,[None])
    def test_external_initial_resume_no_output_commit_refuses_replay(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'7','elsewhere');c['load_ckpt']=7;seen=[]
            def fail(c):seen.append(c['load_ckpt']);raise ValueError('failure')
            with self.assertRaisesRegex(RuntimeError,'replay'):run_attempts(c,fail,complete_steps=lambda p:[])
            self.assertEqual(seen,[7])
    def test_retry_changes_none_to_local_committed_latest(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'none');c['load_ckpt']=None;seen=[]
            def run(c):
                seen.append((c['load_dir'],c['load_ckpt']))
                if len(seen)==1:group(c,5);raise ValueError('failure')
            run_attempts(c,run,complete_steps=lambda p:[5],validate=lambda p,s:None)
            self.assertEqual([x[1] for x in seen],[None,5]);self.assertEqual(c['ckpt'],'latest')
    def test_maxattempts_final_failure_no_extra_retry_message(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,retry=3);c['load_ckpt']=5;group(c,5);seen=[];text=io.StringIO();record={'attempts':[]}
            def fail(c):seen.append(c['load_ckpt']);raise ValueError('failed')
            with contextlib.redirect_stdout(text),self.assertRaisesRegex(RuntimeError,'exhausted 3'):run_attempts(c,fail,record=record,complete_steps=lambda p:[5],validate=lambda p,s:None)
            self.assertEqual(seen,[5,5,5]);self.assertEqual(text.getvalue().count('[Retry]'),2);self.assertEqual(record['status'],'failed')
    def test_zero_retry_means_one_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,retry=0);c['load_ckpt']=None;seen=[]
            def fail(c):seen.append(1);raise ValueError()
            with self.assertRaisesRegex(RuntimeError,'exhausted 1'):run_attempts(c,fail)
            self.assertEqual(seen,[1])
    def test_failure_subprocess_nonzero_cpu_only(self):
        code="from trellis2.utils.train_launch_contract import run_attempts\nc={'output_dir':'unused','load_dir':'unused','ckpt':'none','load_ckpt':None,'auto_retry':0}\ndef fail(c):raise ValueError('stub')\nrun_attempts(c,fail)"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertNotEqual(result.returncode,0);self.assertIn('exhausted 1',result.stderr)
    def test_attempt_record_failure_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            c=cfg(td,'none',retry=0);c['load_ckpt']=None;p,r=preserve_launch_records(c,{},'command',__file__)
            def fail(c):raise ValueError('error')
            with self.assertRaises(RuntimeError):run_attempts(c,fail,p,r)
            data=json.loads(p.read_text());self.assertEqual(data['status'],'failed');self.assertIn('traceback',data['attempts'][0]);self.assertTrue(data['finished_utc'])
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
    def test_fresh_guard_ast_precedes_history_write(self):
        source=Path('train.py').read_text()
        self.assertLess(source.index('cfg = find_ckpt(cfg)'),source.index('record_path, record = preserve_launch_records'))
        self.assertNotIn("with open(os.path.join(cfg.output_dir, 'config.json'), 'w')",source)
        self.assertNotIn("with open(os.path.join(cfg.output_dir, 'command.txt'), 'w')",source)

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
    def test_actual_ast_launch_final_failure_propagates_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);conf=root/'input.json';conf.write_text(json.dumps({'models':{'denoiser':{}},'trainer':{'args':{}}}));out=root/'fresh';calls=[]
            def stub(rank,cfg):calls.append(cfg.load_ckpt);raise ValueError('CPU stub failure')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'exhausted 1'):
                self.execute_launch_block(['--config',str(conf),'--output_dir',str(out),'--ckpt','none','--num_gpus','1','--auto_retry','0'],stub)
            self.assertEqual(calls,[None]);logs=list((out/'run_attempts').glob('*.json'));self.assertEqual(len(logs),1)
            self.assertEqual(json.loads(logs[0].read_text())['status'],'failed')
    def test_actual_ast_launch_guard_preserves_old_history_no_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);conf=root/'input.json';conf.write_text(json.dumps({'models':{'denoiser':{}},'trainer':{'args':{}}}));out=root/'existing';out.mkdir();(out/'config.json').write_bytes(b'original config');(out/'command.txt').write_bytes(b'original command')
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()};calls=[]
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'overwrite'):
                self.execute_launch_block(['--config',str(conf),'--output_dir',str(out),'--ckpt','none','--num_gpus','1'],lambda r,c:calls.append(c))
            self.assertEqual(calls,[]);self.assertEqual({p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()},before)

    def test_train_ast_findckpt_delegates(self):
        source=Path('train.py').read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='find_ckpt');space={'resolve_checkpoint':lambda c:dict(c,passed=True)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'train.py','exec'),space)
        self.assertTrue(space['find_ckpt']({})['passed']);self.assertNotIn('misc_*.pt',source)
    def test_pretrained_rope_preserved_and_bad_keys_fail(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.register_buffer('rope_phases',torch.tensor([1j]))
        model=Tiny();load_pretrained_checked(model,{'weight':torch.ones(1),'rope_phases':torch.zeros(1)});self.assertEqual(model.rope_phases.item(),1j)
        with self.assertRaises(RuntimeError):load_pretrained_checked(model,{'other':torch.ones(1)})

if __name__=='__main__':unittest.main(verbosity=2)
