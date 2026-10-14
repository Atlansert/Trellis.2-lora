import os,time,json,hashlib,pickle,csv,argparse,concurrent.futures,collections,struct,re
from pathlib import Path
import numpy as np
ROOT=Path('/data/lzx/TRELLIS.2');DATA=ROOT/'training_data/multiview_1186';OUT=ROOT/'diagnostics/quality_followup_20261013/geometry_asset_scan'
ap=argparse.ArgumentParser();ap.add_argument('--budget',type=float,default=500);a=ap.parse_args();started=time.monotonic();OUT.mkdir(exist_ok=True);chunks=OUT/'chunks';chunks.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def atomic(p,obj):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
contract=ROOT/'diagnostics/remediation_20261013/canonical_contract_v2_native_staged/manifest.json';mb=contract.read_bytes();objects=[x['object'] for x in json.loads(mb)['objects'] if x['admitted']];assert len(objects)==548
metadata=DATA/'metadata.csv';source={x['sha256']:str((ROOT/x['local_path']).resolve()) for x in csv.DictReader(metadata.open())}
manifest={'schema':'trellis.geometry-risk-scan.v1','objects':objects,'contract_sha256':sha(mb),'metadata_sha256':sha(metadata.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),'thresholds':{'max_triangle_area_normalized_gt':.005,'max_face_axis_span_fraction_gt':.5,'vertex_pca_thin_ratio_lt':.02},'cpu_threads':4,'face_batch':65536,'material_limit':'mesh_dumps have no material; risk flags never true corrupt verdict','training_approved':False}
mp=OUT/'manifest.json'
if mp.exists():assert json.loads(mp.read_text())==manifest,'resume manifest changed'
else:atomic(mp,manifest)
def scan(obj):
 rec={'object':obj,'source_asset':source.get(obj),'flags':[]}
 try:
  p=DATA/'mesh_dumps'/f'{obj}.pickle';raw=p.read_bytes();rec.update(dump_path=str(p),dump_sha256=sha(raw),dump_bytes=len(raw));parts=pickle.loads(raw)['objects'];del raw
  vv=[np.asarray(x['vertices']) for x in parts if len(x['vertices'])];lo=np.min([v.min(0) for v in vv],0);hi=np.max([v.max(0) for v in vv],0);e=hi-lo;scale=float(e.max());assert scale>0
  rec['spans_native']=e.tolist();rec['normalization_longest_span']=scale;total=0.;maxarea=0.;maxratio=0.;maxface=None;nf=0;nv=0;deg=0;big=0;longface=0;normhist=np.zeros(30,np.int64);samples=[]
  for part in parts:
   v=np.asarray(part['vertices']);f=np.asarray(part['faces']);nv+=len(v);nf+=len(f);assert np.isfinite(v).all();assert f.ndim==2 and f.shape[1]==3 and f.min()>=0 and f.max()<len(v)
   samples.append(v[np.linspace(0,len(v)-1,min(len(v),50000),dtype=int)].astype(np.float64))
   for j in range(0,len(f),65536):
    tri=v[f[j:j+65536]].astype(np.float64);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n=np.linalg.norm(cross,axis=1);ar=n/2.;total+=float(ar.sum());idx=int(ar.argmax())
    if ar[idx]>maxarea:maxarea=float(ar[idx]);maxface=tri[idx].tolist()
    deg+=int((ar<scale*scale*1e-14).sum());big+=int((ar/(scale*scale)>.005).sum());ratio=np.ptp(tri,axis=1)/np.maximum(e,scale*1e-9);maxratio=max(maxratio,float(ratio.max()));longface+=int((ratio.max(1)>.5).sum())
    unit=np.abs(cross[n>0]/n[n>0,None]);
    for axis in range(3):normhist[axis*10:(axis+1)*10]+=np.histogram(unit[:,axis],bins=np.linspace(0,1,11))[0]
  s=np.concatenate(samples,0);ev=np.linalg.eigvalsh(np.cov(s,rowvar=False));thin=float(np.sqrt(max(ev[0],0)/max(ev[-1],1e-30)));rec.update(vertices=nv,faces=nf,surface_area_normalized=total/(scale*scale),max_triangle_area_normalized=maxarea/(scale*scale),max_triangle_fraction_total_area=maxarea/total if total else None,max_face_axis_span_fraction=maxratio,large_triangles=big,long_span_faces=longface,degenerate_faces=deg,vertex_sample_pca_eigenvalues=ev.tolist(),vertex_pca_thin_ratio=thin,absolute_face_normal_histogram_10bins_per_axis=normhist.tolist(),largest_triangle_native=maxface)
  if rec['max_triangle_area_normalized']>.005:rec['flags'].append('large_triangle_area')
  if maxratio>.5:rec['flags'].append('large_face_span')
  if thin<.02:rec['flags'].append('thin_geometry_vertex_proxy')
 except Exception as ex:rec['error']=f'{type(ex).__name__}: {ex}'
 return rec
records=[];stopped=False
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for start in range(0,len(objects),32):
  path=chunks/f'chunk_{start//32:03d}.json';hp=path.with_suffix('.sha256.json')
  if path.exists():
   assert hp.exists() and sha(path.read_bytes())==json.loads(hp.read_text())['sha256'];batch=json.loads(path.read_text());assert [x['object'] for x in batch]==objects[start:start+32];records.extend(batch);continue
  if time.monotonic()-started>=a.budget:stopped=True;break
  batch=list(pool.map(scan,objects[start:start+32]));atomic(path,batch);atomic(hp,{'sha256':sha(path.read_bytes()),'objects':len(batch)});records.extend(batch);print('CHUNK',start//32,'objects',len(records),'elapsed',time.monotonic()-started,flush=True)
if len(records)!=len(objects):print('BUDGET_STOP',len(records),flush=True);raise SystemExit(0)
valid=[x for x in records if 'error' not in x];flagged=sorted([x for x in valid if x['flags']],key=lambda x:x['max_triangle_area_normalized'],reverse=True)
# Read only top candidate GLB JSON chunk, never texture buffers.
for rec in flagged[:40]:
 try:
  p=Path(rec['source_asset'])
  if p.suffix.lower()=='.glb':
   with p.open('rb') as f:
    header=f.read(12);magic,version,length=struct.unpack('<III',header);assert magic==0x46546c67;size,kind=struct.unpack('<II',f.read(8));assert kind==0x4e4f534a and size<32*1024*1024;g=json.loads(f.read(size))
   rec['glb_json_review']={'node_names':[x.get('name','') for x in g.get('nodes',[])],'mesh_names':[x.get('name','') for x in g.get('meshes',[])],'materials':[{'name':x.get('name'),'alphaMode':x.get('alphaMode','OPAQUE'),'doubleSided':x.get('doubleSided',False)} for x in g.get('materials',[])],'no_texture_buffers_read':True}
 except Exception as ex:rec['glb_json_error']=f'{type(ex).__name__}: {ex}'
summary={'objects':len(records),'read_errors':len(records)-len(valid),'flagged_objects':len(flagged),'flag_counts':dict(collections.Counter(f for x in valid for f in x['flags'])),'max_triangle_area_normalized_quantiles':np.percentile([x['max_triangle_area_normalized'] for x in valid],[0,25,50,75,90,95,99,100]).tolist(),'max_face_span_fraction_quantiles':np.percentile([x['max_face_axis_span_fraction'] for x in valid],[0,25,50,75,90,95,99,100]).tolist(),'known_examples':{obj:next(x for x in records if x['object']==obj) for obj in ['0086bbab37559442','0fab4af3f9a73bdd']},'elapsed_seconds':time.monotonic()-started,'material_review_scope':'top40flagged GLB JSON only; textures not read','limitations':'geometry proxies only; large triangle may be legitimate panel, thin geometry may be real. Dumps have vertices/faces only, cannot decide visibility/alpha. Flags must be material reviewed, never auto-remove or training approve. Resume checks chunk hashes but does not reread unchanged past inputs; each recorded dump SHA identifies read contents.','training_approved':False}
atomic(OUT/'records.json',records);atomic(OUT/'risk_candidates.json',flagged);atomic(OUT/'summary.json',summary);atomic(OUT/'COMPLETE.json',{'objects':len(records),'records_sha256':sha((OUT/'records.json').read_bytes()),'risk_sha256':sha((OUT/'risk_candidates.json').read_bytes()),'summary_sha256':sha((OUT/'summary.json').read_bytes()),'manifest_sha256':sha(mp.read_bytes()),'training_approved':False});print(json.dumps({k:v for k,v in summary.items() if k!='known_examples'},indent=2),flush=True)
