"""CPU metadata-only immutable next-frame candidate; never ready targets."""
import csv,json,hashlib,struct,math,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/data/lzx/TRELLIS.2');B=ROOT/'diagnostics/quality_followup_20261013';OUT=B/'common_frame_cohort_candidate'
OLD=ROOT/'diagnostics/remediation_20261013/canonical_contract_v2_native_staged/manifest.json'
QUAR={'0fab4af3f9a73bdd':'confirmed_reference_photo_cards','f50d6a143bdfe293':'confirmed_reference_photo_cards','5d4a5e8e46991642':'confirmed_reference_photo_cards','164d20b761652694':'geometry_proxy_uncertain','3d80f1c0d8bebe55':'geometry_proxy_uncertain'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def atomic(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(t,p)
def glbmeta(p):
 with open(p,'rb') as f:
  header=f.read(12);magic,version,total=struct.unpack('<4sII',header)
  if magic!=b'glTF' or version!=2 or total!=p.stat().st_size:raise ValueError('invalid GLB header')
  length,kind=struct.unpack('<I4s',f.read(8))
  if kind!=b'JSON' or length>32*1024*1024:raise ValueError('invalid/oversize GLB JSON chunk')
  g=json.loads(f.read(length))
 meshes=g.get('meshes',[]);nodes=g.get('nodes',[]);materials=g.get('materials',[]);accessors=g.get('accessors',[])
 primitive=[]
 for mi,m in enumerate(meshes):
  for pi,q in enumerate(m.get('primitives',[])):
   index=q.get('indices');count=accessors[index].get('count') if index is not None and index<len(accessors) else None
   primitive.append({'mesh_index':mi,'primitive_index':pi,'mode':q.get('mode',4),'index_count':count,'triangle_count_metadata':count//3 if count is not None and q.get('mode',4)==4 else None,'material':q.get('material')})
 return {'scene':g.get('scene'),'scene_count':len(g.get('scenes',[])),'nodes':[{'index':i,'name':n.get('name'),'mesh':n.get('mesh'),'children':n.get('children',[]),'matrix':n.get('matrix'),'translation':n.get('translation'),'rotation':n.get('rotation'),'scale':n.get('scale')} for i,n in enumerate(nodes)],'meshes':[{'index':i,'name':m.get('name')} for i,m in enumerate(meshes)],'primitives':primitive,'materials':[{'index':i,'name':m.get('name'),'alphaMode':m.get('alphaMode','OPAQUE'),'doubleSided':m.get('doubleSided',False),'baseColorTexture':m.get('pbrMetallicRoughness',{}).get('baseColorTexture')} for i,m in enumerate(materials)],'buffers_metadata':[{'byteLength':x.get('byteLength'),'uri':x.get('uri')} for x in g.get('buffers',[])],'positions_or_textures_decoded':False}
start=time.monotonic();OUT.mkdir(exist_ok=False);(OUT/'chunks').mkdir()
old=json.loads(OLD.read_text());root=Path(old['root']);metadata=root/'metadata.csv'
assert sha(metadata)==old['metadata_sha256']
with open(metadata,newline='') as f:rows={x['sha256']:x for x in csv.DictReader(f)}
admitted=[x for x in old['objects'] if x['admitted']];assert len(admitted)==548 and set(QUAR).issubset({x['object'] for x in admitted})
inputs={'old_candidate_manifest':{'path':str(OLD),'sha256':sha(OLD)},'metadata':{'path':str(metadata),'sha256':sha(metadata)},'geometry_risks':{'path':str(B/'geometry_asset_scan/risk_candidates.json'),'sha256':sha(B/'geometry_asset_scan/risk_candidates.json')}}
def one(o):
 obj=o['object'];r={'object':obj,'status':'quarantined' if obj in QUAR else 'provisional_source_candidate','quarantine_reason':QUAR.get(obj),'training_approved':False,'ready_target':False,'new_shape_latent':None,'new_ss_mean':None,'nose_policy':'preserve_selected_existing_side; no cross-asset nose sign normalization','required_steps':['source_review','proper_mesh_transform','1024_dualgrid_shape_encode','SS_encode_from_shape64_coords','roundtrip','conditional_semantic_assessment']}
 try:
  row=rows[obj];rel=Path(row['local_path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source metadata path')
  source=ROOT/rel;r['source']={'metadata_local_path':str(rel),'resolved_path':str(source.resolve()),'format':source.suffix.lower(),'size_bytes':source.stat().st_size,'sha256':sha(source)}
  if source.suffix.lower()=='.glb':r['source_glb_metadata']=glbmeta(source)
  else:r['status']='unsupported_native_up_requires_calibration' if obj not in QUAR else r['status']
  rr=root/'renders_cond_t'/obj;tf=rr/'transforms.json'
  if not rr.is_symlink():raise ValueError('native alpha symlink origin changed')
  if sha(tf)!=o['transforms_sha256']:raise ValueError('transforms hash changed')
  frames=json.loads(tf.read_text())['frames'];r['frames']={'path':str(tf),'sha256':o['transforms_sha256'],'matrices_are_historical_placeholders':True,'metadata_num_views':row.get('num_views'),'actual_frame_count':len(frames),'accepted_inputs':[]}
  byview={v['view']:v for v in o['views']}
  for i in o['input_views']:
   fr=frames[i];v=byview[i];fp=Path(fr['file_path'])
   if fp.is_absolute() or '..' in fp.parts or fr['file_path']!=v['file'] or fr.get('is_real',False):raise ValueError('invalid accepted frame')
   yaw=float(fr['yaw_deg'])
   if not math.isfinite(yaw) or abs(yaw-i*15)>1e-6:raise ValueError('accepted yaw sequence changed')
   ip=rr/fp
   if sha(ip)!=v['image_sha256']:raise ValueError('accepted image hash changed')
   r['frames']['accepted_inputs'].append({'view':i,'file':fr['file_path'],'yaw_degrees':yaw,'image_path':str(ip),'image_sha256':v['image_sha256'],'legacy_PCA_angle_degrees':v['angle_degrees'],'legacy_PCA_is_not_nose_anchor':True})
  target=o['candidate_target_view'];assert target in o['input_views'];yaw=float(frames[target]['yaw_deg']);angle=math.radians(yaw);c,s=math.cos(angle),math.sin(angle)
  # Row vector native→(x,-z,y), then row Rz(yaw).
  matrix=[[c,-s,0],[0,0,1],[-s,-c,0]]
  r['planned_frame']={'schema':'trellis.common-pipeline-frame.v2','source_convention':'native_glTF_Y_up_assumed_pending_asset_semantic_review' if source.suffix.lower()=='.glb' else 'unknown_OBJ_up_no_transform_approved','raw_axes':['depth','length','up'],'raw_length_axis':1,'raw_up_axis':2,'formal_GLb_transform':'(x,z,-y)','selected_source_view':target,'selected_yaw_degrees':yaw,'matrix_row_native_to_raw':matrix if source.suffix.lower()=='.glb' else None,'determinant':1 if source.suffix.lower()=='.glb' else None,'legacy_voxel_permutation_applied':False,'geometry_reencoded':False,'source_ownership':'user project local asset metadata; rights/provenance not independently certified'}
  r['legacy_targets_are_not_ready_new_targets']=True
 except Exception as e:r['status']='error';r['error']=f'{type(e).__name__}: {e}'
 return r
records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for j in range(0,len(admitted),32):
  chunk=list(pool.map(one,admitted[j:j+32]));records.extend(chunk);p=OUT/'chunks'/f'{j//32:03d}.json';atomic(p,chunk);atomic(p.with_suffix('.sha256.json'),{'sha256':sha(p),'objects':len(chunk)})
  print('METADATA_COHORT',len(records),'elapsed',round(time.monotonic()-start,2),flush=True)
  if time.monotonic()-start>480 and len(records)<len(admitted):raise RuntimeError('480sec budget stop; chunks preserved, no COMPLETE')
from collections import Counter
formats=Counter(x.get('source',{}).get('format','missing') for x in records);status=Counter(x['status'] for x in records)
summary={'input_admitted_objects':548,'known_risks_quarantined':5,'candidate_unflagged_objects':543,'source_format_counts_all':dict(formats),'candidate_format_counts':dict(Counter(x.get('source',{}).get('format','missing') for x in records if x['object'] not in QUAR)),'status_counts':dict(status),'source_hashes_computed':sum('sha256' in x.get('source',{}) for x in records),'accepted_images_verified':sum(len(x.get('frames',{}).get('accepted_inputs',[])) for x in records),'eligible_GLb_provisional':sum(x['status']=='provisional_source_candidate' and x.get('source',{}).get('format')=='.glb' for x in records),'ready_targets':0,'training_approved':False,'errors':[{'object':x['object'],'error':x['error']} for x in records if x['status']=='error'],'warning':'Unflagged means no recorded geometry proxy flag, not clean geometry/semantics or training readiness; source native Y-up and target nose require review. Old means/features are not linked as new ready targets.','elapsed_seconds':time.monotonic()-start}
manifest={'schema':'trellis.common-frame-cohort-candidate.v2','training_approved':False,'status':'metadata_plan_only','inputs':inputs,'source_script_sha256':sha(Path(__file__)),'planned_pipeline_frame':'raw(depth,length,up); lengthY/upZ; formal lengthZ/upY','quarantine_policy':QUAR,'objects':records}
atomic(OUT/'manifest.json',manifest);atomic(OUT/'summary.json',summary)
assert len(records)==548 and not summary['errors'];assert summary['ready_targets']==0
atomic(OUT/'COMPLETE.json',{'manifest_sha256':sha(OUT/'manifest.json'),'summary_sha256':sha(OUT/'summary.json'),'training_approved':False,'metadata_candidate_complete':True});print(json.dumps(summary,indent=2),flush=True)
(OUT/'REPORT.md').write_text('# Common-frame cohort metadata candidate\n\nCPU source JSON/header/hash and accepted-input/frame verification only. No positions, textures or latents decoded; no GPU, reencoding, training, cleanup or original files changed.\n\n'+json.dumps(summary,indent=2)+'\n\nAll entries ready_target=false; new_shape_latent/new_ss_mean null. Required source/semantic reviews and actual geometry/VAE reencoding remain outstanding. Five known risk assets are quarantined (3 confirmed reference cards,2 uncertain proxies), not automatically deleted or all called corrupt. For GLB only, planned proper native-to-common matrix uses selected accepted-view yaw and preserves that side; no universal nose sign assertion. OBJ orientation cannot default to glTF Y-up. Historical num_views and placeholder camera matrices are retained with explicit limitations.\n',encoding='utf-8')
print('COHORT_COMPLETE_NO_TRAINING',flush=True)
