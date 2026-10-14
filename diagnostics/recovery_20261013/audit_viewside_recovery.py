"""Read-only recovery audit; never modifies project sources or training data."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument('--project', required=True)
ap.add_argument('--output', required=True)
a = ap.parse_args()
root = Path(a.project).resolve()
out = Path(a.output).resolve()
out.mkdir(exist_ok=False)
os.chdir(root)
sys.path.insert(0, str(root))
torch.set_num_threads(4)
spec = importlib.util.spec_from_file_location('recovery_ssviews', root/'trellis2/datasets/sparse_structure_latent_views.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
r = root/'training_data/multiview_1186'
rr = r/'renders_cond_t'
roots = {'train_data': {'base': str(r), 'ss_view_latent': str(r/'ss_latents_views_canon'), 'render_cond': str(rr)}}
config = json.loads((root/'results/multiview_1186_ss_flow_lora_64_views_side/config.json').read_text())
dataset = module.ImageConditionedSparseStructureLatentViews(roots, **config['dataset']['args'])
cache = json.loads((rr/'_view_offaxis.json').read_text())
aspects = json.loads((rr/'_view_aspect.json').read_text())
records = []
for _, obj, od, valid in dataset.instances:
    frames = json.loads((rr/obj/'transforms.json').read_text())['frames']
    for i in valid:
        name = Path(frames[i]['file_path']).name
        ang = cache.get(obj, {}).get(name)
        records.append({'object': obj, 'view': i, 'image': str(rr/obj/frames[i]['file_path']),
                        'latent': str(Path(od)/f'{i:03d}.npz'), 'angle': ang,
                        'aspect': aspects.get(obj, {}).get(name),
                        'weight': 1/len(dataset)/len(valid)})
angles = np.asarray([float(x['angle']) for x in records if x['angle'] is not None])
report = {'project': str(root), 'objects': len(dataset), 'pairs': len(records),
          'missing_angles': len(records)-len(angles),
          'cache_sha256': hashlib.sha256((rr/'_view_offaxis.json').read_bytes()).hexdigest(),
          'cache_provenance': 'Generator not located; compare independently reconstructed voxel PCA, not proof of provenance.',
          'angle_formula_checked': 'degrees(acos(abs(principal_eigenvector_of_cov(permuted_voxel_coords)[1])))',
          'angle_quantiles': np.percentile(angles,[0,10,25,50,75,90,100]).tolist(),
          'thresholds': {str(t): {'pairs_gt': int(np.sum(angles>t)), 'fraction_gt': float(np.mean(angles>t)),
                                 'object_balanced_probability_gt': float(sum(x['weight'] for x in records if x['angle'] is not None and float(x['angle'])>t))}
                         for t in [10,15,20,30,45,60]},
          'test_001_mapping': 'No training object named 001; test filename is not an object identity. Mapping unknown.'}
# New audit: verify existing angle cache against the actual training-subset voxels.
errors = []; gaps = []; invalid = []; axis_near = {}
for n,x in enumerate(records):
    p = r/'ss_view_voxels_canon_v4'/x['object']/f"{x['view']:03d}.npz"
    if not p.exists():
        invalid.append({'object':x['object'],'view':x['view'],'issue':'missing voxel'}); continue
    c = np.load(p)['coords'].astype(np.float64)[:,[2,0,1]]
    if len(c)<3:
        invalid.append({'object':x['object'],'view':x['view'],'issue':'too few voxels'}); continue
    vals,vecs = np.linalg.eigh(np.cov(c,rowvar=False))
    ang = float(np.degrees(np.arccos(np.clip(abs(vecs[1,-1]),0,1))))
    gap = float(vals[-1]/max(vals[-2],1e-12))
    gaps.append(gap)
    err = abs(ang-float(x['angle']))
    errors.append(err)
    if err>0.011:
        invalid.append({'object':x['object'],'view':x['view'],'issue':'angle differs','cached':x['angle'],'recomputed':ang})
    if float(x['angle'])<=10:
        axis = int((np.ptp(c,axis=0)+1).argmax())
        axis_near[str(axis)] = axis_near.get(str(axis),0)+1
    if (n+1)%2000==0: print('PCA checked',n+1,flush=True)
report['pca_cache_check'] = {'count':len(errors),'max_absolute_error_degrees':float(max(errors)),
                           'median_error_degrees':float(np.median(errors)),
                           'eigenvalue_ratio_quantiles':np.percentile(gaps,[0,10,50,90,100]).tolist(),
                           'near_side_voxel_argmax_counts':axis_near, 'exceptions':invalid[:100], 'exception_count':len(invalid)}
# Small targeted decoder audit, using CPU and the exact existing frame ordering.
import trellis2.models as models
decpath = '/home/test/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16'
dec = models.from_pretrained(decpath)
dec.convert_to_fp32(); dec.float().eval()
selected_objects = [x[1] for x in dataset.instances[:3]]
samples = []
for obj in selected_objects:
    candidates = [x for x in records if x['object']==obj]
    chosen = []
    for target in [0,30,45]:
        x=min(candidates,key=lambda y:abs(float(y['angle'])-target))
        if x['view'] not in [q['view'] for q in chosen]: chosen.append(x)
    samples.extend(chosen)
canvas=Image.new('RGB',(1200,len(samples)*180),'#181818')
draw=ImageDraw.Draw(canvas)

def paste_mask(mask, origin):
    im=Image.fromarray(np.uint8(mask)*255)
    box=im.getbbox()
    if box: im=im.crop(box)
    im.thumbnail((240,120),Image.Resampling.NEAREST)
    canvas.paste(im.convert('RGB'),(origin[0]+(260-im.width)//2,origin[1]+(140-im.height)//2))

for n,x in enumerate(samples):
    zfile=np.load(x['latent']); z=zfile['mean'] if 'mean' in zfile else zfile['z']
    with torch.inference_mode(): occ=(dec(torch.from_numpy(z).float()[None])>0)[0,0].numpy()
    nz=np.argwhere(occ)
    c=np.load(r/'ss_view_voxels_canon_v4'/x['object']/f"{x['view']:03d}.npz")['coords'].astype(int)[:,[2,0,1]]
    grid=np.zeros((64,64,64),dtype=bool); grid[c[:,0],c[:,1],c[:,2]]=True
    e=(np.ptp(nz,axis=0)+1).tolist() if len(nz) else [0,0,0]
    union=np.logical_or(grid,occ).sum()
    x['decoded_span']=e
    x['voxel_span']=(np.ptp(c,axis=0)+1).tolist()
    x['voxel_decode_iou']=float(np.logical_and(grid,occ).sum()/union) if union else 0.
    x['latent_shape']=list(z.shape)
    im=Image.open(x['image']).convert('RGBA')
    alpha=np.asarray(im.getchannel('A'))>128
    ys,xs=np.nonzero(alpha)
    x['full_resolution_alpha_aspect']=float((xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)) if len(xs) else 0.
    canvas.paste(ImageOps.contain(im.convert('RGB'),(240,120)),(0,n*180+30))
    paste_mask(alpha,(280,n*180+25))
    paste_mask(np.any(grid,axis=2)[::-1,:],(560,n*180+25))
    paste_mask(np.any(occ,axis=2)[::-1,:],(840,n*180+25))
    draw.text((0,n*180),f"{x['object']} view {x['view']:03d} offaxis {x['angle']} deg  spans voxel {x['voxel_span']} decoded {e}",fill='white')
    for label,col in [('render',0),('render alpha',280),('voxel projection',560),('decoded latent projection',840)]: draw.text((col,n*180+155),label,fill='white')
    print('DECODED',x['object'],x['view'],e,flush=True)
report['targeted_samples']=samples
report['interpretation_limits']=['PCA agreement establishes numeric meaning on the checked subset, not the missing original generator.',
 'Silhouettes are orthographic proxies; not a handedness or exact perspective proof.',
 'Three sample objects do not establish all latents are paired correctly.',
 'Current dataset reconstructs the logged count; historical immutable sample manifest is absent.',
 'Offaxis prevalence does not prove causation of optimization collapse.']
(out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'training_subset.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
canvas.save(out/'pairing_samples.png')
print(json.dumps({k:v for k,v in report.items() if k!='targeted_samples'},indent=2),flush=True)
print('OUTPUT',out,flush=True)
