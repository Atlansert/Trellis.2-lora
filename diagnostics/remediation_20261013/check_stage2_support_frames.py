"""Stage2 diagnostic, run only after data-stage risk coverage; no axis correction.
Compare existing canonical voxel support to shape training coords at common64.
24/48 signed axis maps are diagnostic hypotheses, never geometry fixes.
"""
import argparse,itertools,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--shape-name',default='shape_enc_next_dc_f16c32_fp16_1024');p.add_argument('--output',required=True);a=p.parse_args()
contract=Path(a.contract);manifest=json.loads((contract/'manifest.json').read_text());root=Path(manifest['root']);out=Path(a.output)
if out.exists():raise FileExistsError(out)
complete=json.loads((contract/'COMPLETE.json').read_text())
assert complete['manifest_sha256']==hashlib.sha256((contract/'manifest.json').read_bytes()).hexdigest()
perms=list(itertools.permutations(range(3)));signs=list(itertools.product([1,-1],repeat=3));maps=[(perm,sign) for perm in perms for sign in signs]
def keys(c):
 c=np.asarray(c,dtype=np.int64);assert c.ndim==2 and c.shape[1]==3 and c.min()>=0 and c.max()<64
 return np.unique(c[:,0]*4096+c[:,1]*64+c[:,2])
def iou(x,y):
 inter=np.intersect1d(x,y,assume_unique=True).size;return inter/(len(x)+len(y)-inter)
records=[]
for ob in manifest['objects']:
 if not ob['admitted']:continue
 obj=ob['object'];vp=root/'ss_view_voxels_canon_v4'/obj/f"{ob['candidate_target_view']:03d}.npz";sp=root/'shape_latents'/a.shape_name/f'{obj}.npz'
 if not sp.exists():records.append({'object':obj,'error':'shape_latent_missing'});continue
 with np.load(vp,allow_pickle=False) as z:ss=z['coords'][:,[2,0,1]].astype(np.int64)
 with np.load(sp,allow_pickle=False) as z:shape=z['coords'].astype(np.int64)
 ks=keys(shape);ident=iou(keys(ss),ks);scores=[]
 for perm,sign in maps:
  transformed=ss[:,perm].copy()
  for j,s in enumerate(sign):
   if s<0:transformed[:,j]=63-transformed[:,j]
  scores.append(iou(keys(transformed),ks))
 best=max(scores);best_idx=[i for i,v in enumerate(scores) if abs(v-best)<1e-12]
 records.append({'object':obj,'source_view':ob['candidate_target_view'],'ss_spans':(np.ptp(ss,axis=0)+1).tolist(),'shape_spans':(np.ptp(shape,axis=0)+1).tolist(),'ss_long_axis':int(np.argmax(np.ptp(ss,axis=0))),'shape_long_axis':int(np.argmax(np.ptp(shape,axis=0))),'ss_tokens':len(keys(ss)),'shape_tokens':len(ks),'identity_iou':ident,'best_signedperm_iou':best,'best_maps':[{'perm':maps[i][0],'sign':maps[i][1],'det':round(np.linalg.det(np.eye(3)[:,maps[i][0]]@np.diag(maps[i][1])))} for i in best_idx],'voxel_sha256':hashlib.sha256(vp.read_bytes()).hexdigest(),'shape_sha256':hashlib.sha256(sp.read_bytes()).hexdigest()})
 if len(records)%100==0:print('stage2 checked',len(records),flush=True)
valid=[x for x in records if 'error' not in x]
result={'method':'CPU support comparison, same64grid, existing v4 perm(2,0,1) vs shape latent coords; grid/hash checked; no generated mesh, no corrections applied; exact support IoU not expected for distinct voxelizers',
        'objects':len(records),'valid':len(valid),'shape_longaxis_counts':{str(i):sum(x['shape_long_axis']==i for x in valid) for i in range(3)},
        'identity_iou_quantiles':np.percentile([x['identity_iou'] for x in valid],[0,25,50,75,100]).tolist(),
        'best_map_iou_quantiles':np.percentile([x['best_signedperm_iou'] for x in valid],[0,25,50,75,100]).tolist(),
        'records':records,'training_approved':False}
out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
