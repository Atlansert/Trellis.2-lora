"""New-data stages only: process-group bounded execution with durable evidence.
No retries/cleanup of existing outputs; CUDA prohibited unless explicitly6/7.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--command',required=True);p.add_argument('--record-prefix',required=True);p.add_argument('--timeout',type=int,default=600);a=p.parse_args()
if not 0<a.timeout<=1200:raise ValueError('finite stagebudget1..1200sec')
spec=json.loads(Path(a.command).read_text());cmd=spec['argv']
if not isinstance(cmd,list) or not cmd or any(not isinstance(x,str) for x in cmd):raise ValueError('explicit argv list required')
mode=spec.get('device','cpu')
if mode not in ['cpu','cuda6','cuda7']:raise ValueError('only CPU/GPU6/GPU7 stage devices permitted')
prefix=Path(a.record_prefix).resolve();prefix.parent.mkdir(parents=True,exist_ok=True)
for ext in ['.log','.pid','.result.json','.exit_code.txt','.command.json']:
 if Path(str(prefix)+ext).exists():raise FileExistsError('stage record already exists; use a new version')
for rec in spec.get('source_artifacts',[]):
 path=Path(rec['path']);h=hashlib.sha256(path.read_bytes()).hexdigest()
 if h!=rec['sha256']:raise ValueError('stage input changed')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if mode=='cpu' else mode[-1],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
Path(str(prefix)+'.command.json').write_text(json.dumps({'argv':cmd,'device':mode,'timeout_seconds':a.timeout,'command_spec_sha256':hashlib.sha256(Path(a.command).read_bytes()).hexdigest()},indent=2));start=time.monotonic();timeout=False
with open(str(prefix)+'.log','xb') as log:
 worker=subprocess.Popen(cmd,cwd=spec.get('cwd','/data/lzx/TRELLIS.2'),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 Path(str(prefix)+'.pid').write_text(str(worker.pid))
 try:code=worker.wait(timeout=a.timeout)
 except subprocess.TimeoutExpired:
  timeout=True;assert os.getpgid(worker.pid)==worker.pid;os.killpg(worker.pid,signal.SIGTERM)
  try:worker.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
  code=124
result={'exit_code':code,'worker_returncode':worker.returncode,'worker_pid':worker.pid,'worker_ended':worker.poll() is not None,'timed_out':timeout,'elapsed_seconds':time.monotonic()-start,'device':mode,'automatic_retry':False}
Path(str(prefix)+'.exit_code.txt').write_text(str(code));Path(str(prefix)+'.result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);sys.exit(code)
