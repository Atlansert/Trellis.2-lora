"""CPU-only restartable alpha audit. Never writes input data; flags are risks, not truth masks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import argparse,json,hashlib,time,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw
VERSION='alpha-risk-v1'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(t,p)
def stats(m):
 y,x=np.nonzero(m); area=int(len(x))
 if not area:return {'area':0,'fraction':0.,'bbox':None,'bbox_fill':0.,'aspect':None,'components':0}
 b=[int(x.min()),int(y.min()),int(x.max()),int(y.max())];w=b[2]-b[0]+1;h=b[3]-b[1]+1
 return {'area':area,'fraction':area/m.size,'bbox':b,'bbox_fill':area/(w*h),'aspect':w/h,'components':int(ndimage.label(m)[1])}
def one(job):
 root,obj=job;root=Path(root);rr=root/'renders_cond_t'/obj; derived=not rr.is_symlink(); frames=json.loads((rr/'transforms.json').read_text())['frames'][:24]
 cache=json.loads((root/'renders_cond_t/_view_offaxis.json').read_text()).get(obj,{})
 records=[]
 for i,f in enumerate(frames):
  try:
   p=rr/f['file_path'];raw=root/'renders_cond'/obj/f['file_path'];rgba=np.asarray(Image.open(p).convert('RGBA'));rgb=rgba[:,:,:3]; a=rgba[:,:,3]; m=a>0;s=stats(m); hi=stats(a>204)
   item={'view':i,'file':f['file_path'],'sha256':digest(p),'offaxis':cache.get(Path(f['file_path']).name),'alpha':s,'alpha_gt204':hi,'alpha_binary':bool(np.all((a==0)|(a==255))),'flags':[]}
   flags=item['flags']
   if not s['area']:flags.append('empty')
   if s['fraction']>=.99:flags.append('near_full_frame')
   if s['fraction']<.002:flags.append('tiny_area_lt_0.2pct')
   if s['bbox_fill']<.15:flags.append('sparse_bbox_lt_0.15')
   if derived:
    h,w=rgb.shape[:2];k=max(2,min(h,w)//64);rf=rgb.astype(np.float32);ring=np.concatenate([rf[:k].reshape(-1,3),rf[-k:].reshape(-1,3),rf[:,:k].reshape(-1,3),rf[:,-k:].reshape(-1,3)]);bg=np.median(ring,axis=0)
    diff=np.abs(rf-bg).max(2)>3.;closed=ndimage.binary_closing(diff,structure=np.ones((5,5)),iterations=1);labels,n=ndimage.label(closed)
    sizes=np.bincount(labels.ravel());sizes[0]=0;largest=(labels==int(sizes.argmax())) if n else closed;expected=ndimage.binary_fill_holes(largest)
    removed=closed & ~m;pre=int(closed.sum());loss=int(removed.sum())
    item['derive']={'border_bg':bg.tolist(),'raw_rgbdiff_area':int(diff.sum()),'closed_area':pre,'closed_components':int(n),'largest_component_area':int(largest.sum()),'removed_after_stored_area':loss,'removed_fraction_of_closed':loss/max(pre,1),'stored_extra_after_fill_area':int((m&~closed).sum()),'mask_reproduced_exactly':bool(np.array_equal(expected,m)),'rgb_matches_original':bool(np.array_equal(rgb,np.asarray(Image.open(raw).convert('RGB')))),'original_sha256':digest(raw)}
    if loss/max(pre,1)>.1:flags.append('removed_components_gt_10pct')
    if not item['derive']['mask_reproduced_exactly']:flags.append('derived_algorithm_mismatch')
    # Only a risk proxy: closed bounding region near background; not ground-truth body pixels.
    bs=stats(closed)['bbox']
    if bs:
     x0,y0,x1,y1=bs; near=(~diff[y0:y1+1,x0:x1+1]);item['derive']['near_bg_fraction_in_closed_bbox']=float(near.mean())
   records.append(item)
  except Exception as e:records.append({'view':i,'file':f['file_path'],'error':str(e),'flags':['read_error']})
 return {'object':obj,'derived':derived,'transform_sha256':digest(rr/'transforms.json'),'frames':records}
def sheets(root,out,records):
 choices=[]
 for obj in records:
  for f in obj['frames']:
   if 'error' in f:continue
   angle=f.get('offaxis');valid=angle is not None and 0<=float(angle)<=10
   severity=100*('empty' in f['flags'])+20*('derived_algorithm_mismatch' in f['flags'])+10*('tiny_area_lt_0.2pct' in f['flags'])+4*('removed_components_gt_10pct' in f['flags'])
   if valid and obj['derived']: choices.append((severity+(10 if obj['object']=='3e988013c8136c9b' else 0),obj,f))
 choices.sort(key=lambda x:(-x[0],x[1]['object'],x[2]['view']))
 for page in range(2):
  sel=choices[page*12:(page+1)*12];canvas=Image.new('RGB',(1200,12*155+35),(25,25,25));d=ImageDraw.Draw(canvas);d.text((8,6),'RISK candidates <=10deg inputs: original / stored alpha / actual train crop. Flags NOT ground truth.',fill='white')
  for row,(_,obj,f) in enumerate(sel):
   p=root/'renders_cond_t'/obj['object']/f['file'];im=Image.open(p).convert('RGBA');ar=np.asarray(im);a=ar[:,:,3];rgb=ar[:,:,:3]; y,x=np.nonzero(a)
   if len(x):
    cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2;hs=max(x.max()-x.min(),y.max()-y.min())/2;crop=im.crop((int(cx-hs),int(cy-hs),int(cx+hs),int(cy+hs))).resize((512,512),Image.Resampling.LANCZOS);ca=np.asarray(crop);train=Image.fromarray((ca[:,:,:3].astype(float)*ca[:,:,3:4]/255).astype('uint8'))
   else:train=Image.new('RGB',(512,512),'black')
   ims=[Image.fromarray(rgb),Image.fromarray(a).convert('RGB'),train]
   for col,v in enumerate(ims):v.thumbnail((390,130));canvas.paste(v,(col*400+(400-v.width)//2,35+row*155))
   d.text((5,35+row*155+131),f"{obj['object']} v{f['view']:02d} angle={f['offaxis']} {','.join(f['flags'])}",fill='white')
  canvas.save(out/f'valid_inputs_risk_contact_{page+1}.png')
 atom(out/'contact_selection.json',[{'object':o['object'],'view':f['view'],'score':s,'flags':f['flags']} for s,o,f in choices[:24]])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seconds',type=int,default=500);args=ap.parse_args();root=Path(args.root).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);chunks=out/'chunks';chunks.mkdir(exist_ok=True)
 objs=sorted([r['sha256'] for r in csv.DictReader((root/'metadata.csv').open())]); manifest={'version':VERSION,'objects':objs,'chunk_objects':32,'script_sha256':digest(__file__),'metadata_sha256':digest(root/'metadata.csv'),'offaxis_sha256':digest(root/'renders_cond_t/_view_offaxis.json'),'source_root':str(root),'method':'flags are risk proxies, not true body loss; nonempty alpha masks do not establish validity; all first24 frames; derived RGBdiff reproduces TOL3/closing5/LCC/fill'}
 if (out/'manifest.json').exists():
  old=json.loads((out/'manifest.json').read_text())
  if old!=manifest:raise RuntimeError('Manifest changed; use new output/version, not stale resume')
 else:atom(out/'manifest.json',manifest)
 start=time.monotonic()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for k in range(0,len(objs),32):
   cp=chunks/f'chunk_{k//32:03d}.json';mark=chunks/f'chunk_{k//32:03d}.complete.json'
   if cp.exists() and mark.exists():
    mm=json.loads(mark.read_text());assert mm['sha256']==digest(cp);continue
   if time.monotonic()-start>args.seconds:print('BUDGET_STOP clean resumable',flush=True);return
   batch=objs[k:k+32];result=list(pool.map(one,[(str(root),x) for x in batch]));atom(cp,result);atom(mark,{'sha256':digest(cp),'objects':batch,'frames':sum(len(x['frames']) for x in result)});print('CHUNK_DONE',k//32,len(batch),round(time.monotonic()-start,2),flush=True)
 records=[]
 for cp in sorted(chunks.glob('chunk_*.json')):
  if '.complete.' not in cp.name:records+=json.loads(cp.read_text())
 assert len(records)==len(objs)
 fs=[(o,f) for o in records for f in o['frames']];counts=Counter(flag for o,f in fs for flag in f['flags']);groups={}
 for kind in ['derived','symlink']:
  sub=[(o,f) for o,f in fs if o['derived']==(kind=='derived')];groups[kind]={'objects':sum(o['derived']==(kind=='derived') for o in records),'frames':len(sub),'flags':dict(Counter(flag for o,f in sub for flag in f['flags'])),'valid_inputs':sum(f.get('offaxis') is not None and 0<=float(f['offaxis'])<=10 for o,f in sub),'read_errors':sum('error' in f for o,f in sub)}
 candidates=[{'object':o['object'],'derived':o['derived'],'flagged_frames':sum(bool(f['flags']) for f in o['frames']),'valid_flagged_views':[f['view'] for f in o['frames'] if f['flags'] and f.get('offaxis') is not None and 0<=float(f['offaxis'])<=10],'flags':dict(Counter(flag for f in o['frames'] for flag in f['flags']))} for o in records if any(f['flags'] for f in o['frames'])]
 atom(out/'records.json',records);atom(out/'risk_candidates.json',candidates);atom(out/'summary.json',{'objects':len(records),'frames':len(fs),'groups':groups,'flags':dict(counts),'flagged_objects':len(candidates),'derived_rgb_or_algorithm_mismatch':sum(o['derived'] and 'derive' in f and (not f['derive']['rgb_matches_original'] or not f['derive']['mask_reproduced_exactly']) for o,f in fs),'elapsed_last_invocation_sec':time.monotonic()-start,'warning':'Candidates are risk flags, not confirmed corrupt/ground-truth masks.'});sheets(root,out,records)
 atom(out/'COMPLETE.json',{'version':VERSION,'objects':len(records),'frames':len(fs),'summary_sha256':digest(out/'summary.json'),'records_sha256':digest(out/'records.json'),'risk_candidates_sha256':digest(out/'risk_candidates.json'),'manifest_sha256':digest(out/'manifest.json')});print('SCAN_COMPLETE',len(records),len(fs),flush=True)
if __name__=='__main__':main()
