from pathlib import Path
import os,sys,time,json,subprocess,signal,hashlib
root=Path('/data/lzx/TRELLIS.2');d=root/'diagnostics/remediation_20261013';name='shape_flow_compatibility';script=d/('run_'+name+'.py');assert script.is_file();assert not (d/name).exists()
command=['/home/test/miniconda3/envs/trellis2/bin/python','-u',str(script)];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONPATH=str(root))
cmdpath=d/(name+'_command.json');assert not cmdpath.exists();cmdpath.write_text(json.dumps({'command':command,'cuda_visible_devices':'6','timeout_seconds':600,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()},indent=2));start=time.monotonic()
with open(d/(name+'.log'),'x') as log:
 p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (d/(name+'.pid')).write_text(str(p.pid));timeout=False
 try:rc=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  timeout=True;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  rc=124
r={'worker_pid':p.pid,'worker_ended':p.poll() is not None,'return_code':rc,'timed_out':timeout,'elapsed_seconds':time.monotonic()-start};(d/(name+'_result.json')).write_text(json.dumps(r,indent=2));(d/(name+'_exit_code.txt')).write_text(str(rc));print(json.dumps(r,indent=2));sys.exit(rc)
