"""CPU preparation default; fresh100 only after independent target approval."""
import argparse,json,os,sys,random,shutil
from pathlib import Path
import numpy as np
from nose_training_contract import validate,Dataset,fresh,sha
p=argparse.ArgumentParser();p.add_argument('--old-manifest',required=True);p.add_argument('--new-target-manifest',required=True);p.add_argument('--candidate-manifest',required=True);p.add_argument('--baseline-run-contract',required=True);p.add_argument('--approval');p.add_argument('--output');p.add_argument('--execute',action='store_true');a=p.parse_args()
if a.execute and not a.approval:raise ValueError('parent approval required before models')
m,targets,baseline=validate(a.old_manifest,a.new_target_manifest,a.candidate_manifest,a.baseline_run_contract,a.approval)
if not a.execute:print(json.dumps({'validation_passed':True,'models_loaded':False,'training_started':False,'max_steps':100,'production_approved':False}));sys.exit(0)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='6' or not a.output:raise ValueError('GPU6 and fresh output required')
root=Path(m['project_root']);sys.path.insert(0,str(root));os.chdir(root)
if shutil.disk_usage(root).free<32*1024**3:raise ValueError('32GiB budget required')
import torch
from trellis2 import models,trainers
from trellis2.modules.lora import apply_lora_to_linear_layers
from scripts.utils.inference_contract import checked_load,TARGETS
from safetensors.torch import load_file
out=fresh(a.output);seed=m['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
recipe=json.loads(Path(m['dependencies']['recipe']['path']).read_text());model=models.SparseStructureFlowModel(**recipe['models']['denoiser']['args']).cuda();state=load_file(m['dependencies']['base']['path']);checked_load(model,state,allow_rope=True,pop_rope=True);del state
for param in model.parameters():param.requires_grad_(False)
replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.);assert len(replaced)==150
adapters=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert len(adapters)==300 and all(p.dtype==torch.float32 and ('lora_down' in n or 'lora_up' in n) for n,p in adapters)
model.train();dataset=Dataset(a.old_manifest,a.new_target_manifest,a.candidate_manifest,a.baseline_run_contract,a.approval)
args=recipe['trainer']['args'];args.update(m['recipe_overrides']);args['max_steps']=100
if args!=baseline['recipe'] or recipe['models']['denoiser']['args']!=baseline['model_args']:raise ValueError('recipe differs from original100: target only variable required')
contract={'schema':'trellis.nose-target-trial-run.v1','old_manifest_sha256':sha(a.old_manifest),'new_target_manifest_sha256':sha(a.new_target_manifest),'candidate_manifest_sha256':sha(a.candidate_manifest),'baseline_run_contract_sha256':sha(a.baseline_run_contract),'approval_sha256':sha(a.approval),'runner_sha256':sha(__file__),'helper_sha256':sha(Path(__file__).with_name('nose_training_contract.py')),'train_objects':dataset.objects,'internal_holdout_objects':m['internal_holdout_objects'],'phase_steps':100,'seed':seed,'base':m['dependencies']['base'],'recipe':args,'model_args':recipe['models']['denoiser']['args'],'trainable_parameters':sum(p.numel() for _,p in adapters),'production_approved':False,'automatic_continuation':False,'fresh_from_official':True,'checkpoint_steps':[50,100],'condition_images_unchanged':True,'only_changed_variable':'target canonical nose -rawY, proper geometric rotation plus actual VAE encoding; no prediction rotation'}
(out/'pilot_run_contract.json').write_text(json.dumps(contract,indent=2));trainer=trainers.ImageConditionedFlowMatchingCFGTrainer({'denoiser':model},dataset,output_dir=str(out),load_dir=str(out),step=None,**args);trainer.run();print('NOSE_TARGET_TRIAL100_COMPLETE_NOT_PRODUCTION',flush=True)
