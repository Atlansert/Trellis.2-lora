"""Prepare-only by default; no training without independent passed target gate."""
import argparse,json,os,random,sys,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/short_training_prepare')
from six_ss_contract import validate,phase_steps,fresh_output,sha,ManifestDataset
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--approval');p.add_argument('--output');p.add_argument('--execute',action='store_true');p.add_argument('--phase_steps',type=int,default=200);p.add_argument('--duration-approval',required=True);a=p.parse_args()
m,by=validate(a.manifest,a.approval,diagnostic=not a.execute)
duration=json.loads(Path(a.duration_approval).read_text())
if a.phase_steps!=200 or duration.get('phase_steps')!=200 or duration.get('schema')!='trellis.same-cohort-resume200-approval.v1' or duration.get('approved') is not True or not duration.get('diagnostic_only'):raise ValueError('explicit200step diagnostic approval required')
if sha(a.manifest)!=duration['manifest_sha256'] or sha(a.approval)!=duration['original_data_approval_sha256']:raise ValueError('duration input changed')
if sha('/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/ss_short_train4_holdout2_step50/pilot_run_contract.json')!=duration['baseline_run_contract_sha256']:raise ValueError('baseline changed')
sys.path.insert(0,'/data/lzx/TRELLIS.2')
from trellis2.utils.checkpoint_contract import validate_checkpoint
resume_dir=Path('/data/lzx/TRELLIS.2/diagnostics/cohort_training_followup_v1/ss_same_cohort_step100')
validate_checkpoint(resume_dir/'ckpts',100,True)
if sha(resume_dir/'pilot_run_contract.json')!=duration['resume_run_contract_sha256']:raise ValueError('resume contract changed')
if not a.execute:
 print(json.dumps({'mode':'CPU_validate_only','train_objects':m['train_objects'],'internal_holdout_objects':m['internal_holdout_objects'],'training_started':False,'approval_required':True},indent=2));sys.exit(0)
if not a.output:raise ValueError('fresh output required')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise ValueError('single GPU6 only for approved short pilot')
root=Path(m['project_root']);sys.path.insert(0,str(root));os.chdir(root)
if shutil.disk_usage(root).free<32*1024**3:raise ValueError('32GiB conservative free budget required')
# Every identity, approval and target gate above precedes CUDA model construction.
import torch
from trellis2 import models,trainers
from trellis2.modules.lora import apply_lora_to_linear_layers
from scripts.utils.inference_contract import checked_load,TARGETS
from safetensors.torch import load_file
out=fresh_output(a.output);random.seed(m['seed']);np.random.seed(m['seed']);torch.manual_seed(m['seed']);torch.cuda.manual_seed_all(m['seed'])
recipe=json.loads(Path(m['dependencies']['recipe']['path']).read_text());model=models.SparseStructureFlowModel(**recipe['models']['denoiser']['args']).cuda();state=load_file(m['dependencies']['base']['path']);checked_load(model,state,allow_rope=True,pop_rope=True);del state
for parameter in model.parameters():parameter.requires_grad_(False)
replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.);assert len(replaced)==150
adapter=[(n,v) for n,v in model.named_parameters() if v.requires_grad];assert len(adapter)==300 and all(v.dtype==torch.float32 and ('lora_down' in n or 'lora_up' in n) for n,v in adapter)
model.train();dataset=ManifestDataset(a.manifest,a.approval);args=recipe['trainer']['args'];args.update(m['recipe_overrides']);args['max_steps']=200
contract={'manifest_sha256':sha(a.manifest),'approval_sha256':sha(a.approval),'runner_sha256':sha(__file__),'helper_sha256':sha('/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/short_training_prepare/six_ss_contract.py'),'duration_approval_sha256':sha(a.duration_approval),'baseline_run_contract_sha256':duration['baseline_run_contract_sha256'],'train_objects':dataset.objects,'internal_holdout_objects':m['internal_holdout_objects'],'phase_steps':200,'seed':m['seed'],'base':m['dependencies']['base'],'recipe':args,'model_args':recipe['models']['denoiser']['args'],'trainable_parameters':sum(v.numel() for _,v in adapter),'production_approved':False,'automatic_continuation':False,'resume_step':100,'resume_from':str(resume_dir),'resume_trajectory_exact':False}
(out/'pilot_run_contract.json').write_text(json.dumps(contract,indent=2));trainer=trainers.ImageConditionedFlowMatchingCFGTrainer({'denoiser':model},dataset,output_dir=str(out),load_dir=str(resume_dir),step=100,**args);trainer.run();print('FOURTRAIN_TWOHOLDOUT_PHASE200_FINISHED_NOT_PRODUCTION',flush=True)
