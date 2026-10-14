"""Independent staged-target integrity validation and old-source change report."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--old-records',required=True);p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output)
if out.exists():raise FileExistsError(out)
r=Path(args.contract);m=json.load(open(r/'manifest.json'));complete=json.load(open(r/'COMPLETE.json'))
assert sha(r/'manifest.json')==complete['manifest_sha256']
assert not complete['training_approved']
root=Path(m['root']);old={x['object']:x for x in json.load(open(args.old_records))}
assert sha(root/'metadata.csv')==m['metadata_sha256']
assert sha(root/'renders_cond_t/_view_offaxis.json')==m['cache_sha256']
admitted=[x for x in m['objects'] if x['admitted']];changed=[]
for x in admitted:
 assert x['mask_origin']=='native_alpha_symlink'
 assert (root/'renders_cond_t'/x['object']).is_symlink()
 assert sha(root/'renders_cond_t'/x['object']/'transforms.json')==x['transforms_sha256']
 assert x['candidate_target_view'] in x['input_views']
 for v in x['views']:
  assert v['angle_degrees']<=m['max_angle_degrees']
  assert np.argmax(v['spans'])==1 and v['mask_nonempty_nonopaque']
  for path,key in [(root/'renders_cond_t'/x['object']/v['file'],'image_sha256'),(root/'ss_view_voxels_canon_v4'/x['object']/f"{v['view']:03d}.npz",'voxel_sha256'),(root/'ss_latents_views_canon'/x['object']/f"{v['view']:03d}.npz",'latent_sha256')]:
   assert sha(path)==v[key]
 target=r/'canonical'/f"{x['object']}.npz"
 source=root/'ss_latents_views_canon'/x['object']/f"{x['candidate_target_view']:03d}.npz"
 assert sha(target)==x['target_sha256'] and sha(source)==x['target_source_sha256']
 a=np.load(target,allow_pickle=False)['mean'];b=np.load(source,allow_pickle=False)['mean']
 assert np.array_equal(a,b) and a.dtype==np.float32 and a.shape==(8,16,16,16) and np.isfinite(a).all()
 if old[x['object']]['matches']!=[x['candidate_target_view']]:changed.append(x['object'])
 print('verified',x['object'],flush=True) if len(changed) and len(changed)%100==0 else None
bad=[x for x in m['objects'] if old[x['object']]['axis10_inputs'] and old[x['object']]['sources'][0]['angle']>10]
result={'manifest_sha256':sha(r/'manifest.json'),'objects':len(admitted),'input_pairs':sum(len(x['input_views']) for x in admitted),'old131_native_corrected':sum(x['admitted'] for x in bad),'old131_quarantined':sum(not x['admitted'] for x in bad),'changed_targets':len(changed),'changed_objects':changed,'all_source_hashes_and_arrays_verified':True,'training_approved':False}
out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='changed_objects'},indent=2))
