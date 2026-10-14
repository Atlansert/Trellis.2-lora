import subprocess,os,signal,time,json,sys
from pathlib import Path
b=Path('/data/lzx/TRELLIS.2/diagnostics/quality_followup_20261013');cmd=['/home/test/miniconda3/envs/trellis2/bin/python','-u',str(b/'run_predicted_support_shape.py')];(b/'predicted_support_shape_command.json').write_text(json.dumps({'argv':cmd,'timeout':600,'gpu':6},indent=2));start=time.monotonic();timeout=False
with open(b/'predicted_support_shape.log','xb') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'),start_new_session=True);(b/'predicted_support_shape.pid').write_text(str(p.pid))
 try:code=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  timeout=True;assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=124
r={'pid':p.pid,'ended':p.poll() is not None,'exit_code':code,'worker_returncode':p.returncode,'timeout':timeout,'seconds':time.monotonic()-start};(b/'predicted_support_shape_result.json').write_text(json.dumps(r,indent=2));(b/'predicted_support_shape.exit_code').write_text(str(code));print(json.dumps(r));sys.exit(code)
