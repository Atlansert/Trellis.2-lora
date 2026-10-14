import os,json,time,signal,subprocess,hashlib,sys
from pathlib import Path
root=Path('/data/lzx/TRELLIS.2');os.chdir(root)
out=root/'results_infer/axis10_20k_003_remediation_20261013';out.mkdir(exist_ok=False)
cmd=['/home/test/miniconda3/envs/trellis2/bin/python','-u','scripts/infer_finetuned.py','--image','data_test/003.png','--output',str(out.relative_to(root)),'--pipeline_type','1024','--vae_decoder','checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt','--vae_only','--sparse_structure_lora','results/multiview_1186_ss_flow_lora_64_canon_axis10/ckpts/denoiser_ema0.9999_step0020000.pt','--ss_lora_rank','8','--ss_lora_alpha','8','--flow_1024_lora','results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt','--shape_lora_rank','8','--shape_lora_alpha','8','--seed','42','--no_video']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
(out/'command.txt').write_text('CUDA_VISIBLE_DEVICES=4 '+__import__('shlex').join(cmd)+'\n')
(out/'command.json').write_text(json.dumps({'argv':cmd,'timeout_seconds':600,'inference_source_sha256':hashlib.sha256((root/'scripts/infer_finetuned.py').read_bytes()).hexdigest(),'ss_canonicalize':False,'ss_swap_yz':False,'coordinate_training_gate':False,'purpose':'existing checkpoint four-image DIAGNOSTIC, not new-data training'},indent=2))
t=time.perf_counter();timed_out=False
with open(out/'infer.log','wb') as log:
 proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'infer.pid').write_text(str(proc.pid))
 try:rc=proc.wait(timeout=600)
 except subprocess.TimeoutExpired:
  timed_out=True
  # Only the new worker process group, never preexisting services.
  assert os.getpgid(proc.pid)==proc.pid
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  rc=124
manifest_path=out/'run_manifest.json'
if timed_out and manifest_path.is_file():
 manifest=json.loads(manifest_path.read_text())
 manifest.update(status='terminated_by_outer_deadline',outer_timeout_seconds=600,worker_returncode=proc.returncode)
 tmp=out/'run_manifest.json.supervisor.tmp';tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,manifest_path)
report={'exit_code':rc,'timed_out':timed_out,'worker_returncode':proc.returncode,'elapsed_seconds':time.perf_counter()-t,'worker_pid':proc.pid,'worker_ended':proc.poll() is not None,'glb_exists':(out/'003.glb').is_file(),'glb_bytes':(out/'003.glb').stat().st_size if (out/'003.glb').is_file() else 0}
(out/'exit_code.txt').write_text(str(rc)+'\n');(out/'bounded_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
# Return diagnostic result without disguising failures from the caller.
sys.exit(rc)
