import os,sys,time,signal,json,subprocess
from pathlib import Path
root=Path('/data/lzx/TRELLIS.2');os.chdir(root);b=root/'diagnostics/quality_followup_20261013';plan=b/'short_ss_pilot_plan';supervisor=b/'short_ss_pilot_execution';supervisor.mkdir(exist_ok=False)
results=[]
for arm in ['legacy','proper']:
 out=b/f'short_ss_{arm}_50';assert not out.exists()
 cmd=['/home/test/miniconda3/envs/trellis2/bin/python','-u',str(plan/'run_short_ss_pilot.py'),'--manifest',str(plan/'manifest.json'),'--arm',arm,'--approval',str(plan/'approval_0086_diagnostic.json'),'--output',str(out),'--execute','--phase_steps','50']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');(supervisor/f'{arm}.command.json').write_text(json.dumps({'argv':cmd,'gpu':6,'outer_timeout':900},indent=2));t=time.perf_counter();timeout=False
 with open(supervisor/f'{arm}.log','wb') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(supervisor/f'{arm}.pid').write_text(str(proc.pid))
  try:code=proc.wait(timeout=900)
  except subprocess.TimeoutExpired:
   timeout=True;assert os.getpgid(proc.pid)==proc.pid;os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 rec={'arm':arm,'pid':proc.pid,'worker_ended':proc.poll() is not None,'exit_code':code,'worker_returncode':proc.returncode,'timed_out':timeout,'seconds':time.perf_counter()-t,'complete50':(out/'ckpts/complete_step0000050.json').is_file()};results.append(rec);(supervisor/f'{arm}.exit_code.txt').write_text(str(code));(supervisor/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(rec),flush=True)
 if code or not rec['complete50']:sys.exit(code or 1)
print('TWO_ARMS_50_COMMITTED_NO_AUTO_CONTINUATION',flush=True)
