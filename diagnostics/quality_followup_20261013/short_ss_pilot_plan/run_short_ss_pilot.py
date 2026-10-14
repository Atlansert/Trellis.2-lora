"""Parent-approved diagnostic ONLY. Default invocation validates CPU and exits."""
import argparse,json,os,sys,random
from pathlib import Path
import numpy as np
from short_ss_pilot_contract import validate,ManifestDataset,fresh_output,sha,checked_phase_steps
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--arm',choices=['legacy','proper'],required=True);p.add_argument('--approval');p.add_argument('--output');p.add_argument('--execute',action='store_true');p.add_argument('--phase_steps',type=int,choices=[50,100],default=50);args=p.parse_args()
m,objs=validate(args.manifest,args.arm,args.approval,diagnostic_only=not args.execute)
if not args.execute:
 print(json.dumps({'mode':'CPU_validate_only','candidates':len(objs),'training_started':False,'requires_parent_approval':True},indent=2));sys.exit(0)
if not args.output:raise ValueError('fresh output required')
if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('4','5','6','7'):raise ValueError('single explicit GPU4..7 required')
approval=json.loads(Path(args.approval).read_text());steps=checked_phase_steps(args.phase_steps,approval['max_steps'])
ROOT=Path(m['project_root']);sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import torch
from trellis2 import models,trainers
from trellis2.modules.lora import apply_lora_to_linear_layers
from scripts.utils.inference_contract import checked_load,TARGETS
# All approval checks above occur before torch CUDA/model construction.
import shutil
# Four full model+EMA checkpoint pairs across two arms, plus optimizer/cache margin.
if shutil.disk_usage(ROOT).free < 32*1024**3:raise ValueError('at least32GiB free required for two-arm pilot budget')
out=fresh_output(args.output)
seed=m['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
cfg=json.loads(Path(m['recipe']['path']).read_text());assert sha(m['recipe']['path'])==m['recipe']['sha256']
model=models.SparseStructureFlowModel(**cfg['models']['denoiser']['args']).cuda()
from safetensors.torch import load_file
state=load_file(m['base']['path']);checked_load(model,state,allow_rope=True,pop_rope=True);del state
for par in model.parameters():par.requires_grad_(False)
replaced=apply_lora_to_linear_layers(model,TARGETS,rank=8,alpha=8.,dropout=0.)
assert len(replaced)==150
adapters=[(n,par) for n,par in model.named_parameters() if par.requires_grad]
assert len(adapters)==300 and all(par.dtype==torch.float32 and ('lora_down' in n or 'lora_up' in n) for n,par in adapters)
model.train()
dataset=ManifestDataset(args.manifest,args.arm,args.approval)
a=cfg['trainer']['args'];a.update(max_steps=steps,batch_size_per_gpu=2,batch_split=1,i_save=50,i_sample=steps+1,i_log=10,i_print=10,snapshot_batch_size=2,prefetch_data=False)
# Two checkpoints/live+EMA groups; no automatic restart or geometry acceptance.
(out/'pilot_run_contract.json').write_text(json.dumps({'manifest_sha256':sha(args.manifest),'approval_sha256':sha(args.approval),'arm':args.arm,'scope':[o['object'] for o in objs],'phase_steps':steps,'approved_total_steps':approval['max_steps'],'seed':seed,'recipe':a,'model_args':cfg['models']['denoiser']['args'],'base_weight':m['base'],'runner_sha256':sha(__file__),'contract_sha256':sha(Path(__file__).with_name('short_ss_pilot_contract.py')),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'not_production':True},indent=2))
trainer=trainers.ImageConditionedFlowMatchingCFGTrainer({'denoiser':model},dataset,output_dir=str(out),load_dir=str(out),step=None,**a)
trainer.run()
print('DIAGNOSTIC_PILOT_FINISHED_NOT_PRODUCTION',flush=True)
