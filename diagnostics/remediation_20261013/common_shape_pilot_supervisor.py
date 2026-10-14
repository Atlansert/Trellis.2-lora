import os,sys,time,json,signal,subprocess,hashlib
from pathlib import Path
root=Path('/data/lzx/TRELLIS.2');os.chdir(root);out=root/'diagnostics/remediation_20261013/common_shape_pilot';out.mkdir(exist_ok=False)
worker=root/'diagnostics/remediation_20261013/common_shape_pilot_worker.py';t=time.perf_counter();records=[]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',PYTHONPATH=str(root),OMP_NUM_THREADS='8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
for stage in ['encode','decode']:
 budget=min(300,600-(time.perf_counter()-t));assert budget>0
 cmd=['/home/test/miniconda3/envs/trellis2/bin/python','-u',str(worker),stage]
 (out/(stage+'_command.json')).write_text(json.dumps({'argv':cmd,'timeout_seconds':budget,'GPU':'5','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest()},indent=2))
 with open(out/(stage+'.log'),'wb') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);(out/(stage+'.pid')).write_text(str(p.pid));st=time.perf_counter();timeout=False
  try:rc=p.wait(timeout=budget)
  except subprocess.TimeoutExpired:
   timeout=True;assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc=124
 record={'stage':stage,'pid':p.pid,'exit_code':rc,'worker_returncode':p.returncode,'timeout':timeout,'worker_ended':p.poll() is not None,'elapsed_seconds':time.perf_counter()-st};records.append(record);(out/(stage+'_exit_code.txt')).write_text(str(rc));print(json.dumps(record),flush=True)
 if rc:break
manifest={'schema':'trellis.common-shape-pilot.v1','training_approved':False,'stage_records':records,'completed':len(records)==2 and all(x['exit_code']==0 for x in records),'elapsed_seconds':time.perf_counter()-t,'resolution':1024,'asset_scope':['0086bbab37559442','0fab4af3f9a73bdd'],'input_manifest_sha256':hashlib.sha256((root/'diagnostics/remediation_20261013/common_frame_pilot/manifest.json').read_bytes()).hexdigest(),'limitations':['PBR not encoded','Fixed shape20k compatibility untested','Roof/wheel/nose semantic labels do not automatically propagate through VAE','Rotation-only PLY diagnostics not formal GLB acceptance'],'file_hashes':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}}
for name in ['encode_result','decode_result']:
 p=out/(name+'.json')
 if p.exists():manifest[name]=json.loads(p.read_text())
p=out/'manifest.json';p.write_text(json.dumps(manifest,indent=2))
if manifest['completed']:(out/'COMPLETE.json').write_text(json.dumps({'completed':True,'resolution':1024,'training_approved':False,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
print('PILOT_FINAL',manifest['completed'],manifest['elapsed_seconds'],flush=True);sys.exit(0 if manifest['completed'] else 1)
