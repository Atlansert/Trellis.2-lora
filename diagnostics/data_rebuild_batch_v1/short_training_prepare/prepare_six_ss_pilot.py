import argparse,json
from pathlib import Path
from PIL import Image
from six_ss_contract import TRAIN,HOLDOUT,SCHEMA,sha,record,fresh_output,production_condition,finite_mean,validate
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root).resolve();batch=root/'diagnostics/data_rebuild_batch_v1';out=fresh_output(a.output)
enc=batch/'encoded_targets_v1/manifest.json';ec=json.loads(enc.with_name('COMPLETE.json').read_text());em=json.loads(enc.read_text())
assert ec['manifest_sha256']==sha(enc);assert em['schema']=='trellis.render-batch-encoded.v1'
review=json.loads((batch/'asset_review/review.json').read_text());review_by={x['object_id']:x for x in review['selected']};assets={x['object_id']:x for x in em['assets']};assert set(assets)==set(TRAIN+HOLDOUT)
rows=[]
for obj in TRAIN+HOLDOUT:
 rdir=batch/'renders'/f'{obj}_cpu_v1';rmanifest=rdir/'manifest.json';r=json.loads(rmanifest.read_text());assert r['status']=='success';assert assets[obj]['render_manifest_sha256']==sha(rmanifest)
 rgba=rdir/'000.png';assert r['views'][0]['index']==0 and r['views'][0]['sha256']==sha(rgba)
 target=batch/'encoded_targets_v1'/obj/'ss_mean.npz';assert assets[obj]['stages']['ss']['files']['ss_mean.npz']==sha(target);finite_mean(target)
 cp=out/f'{obj}_condition512.png'
 with Image.open(rgba) as im:production_condition(im).save(cp)
 rows.append({'object_id':obj,'split':'train' if obj in TRAIN else 'internal_holdout','view_index':0,'other_views':'geometry/camera audit only, never included in train drawslots','render_rgba':record(rgba),'render_manifest':record(rmanifest),'canonical_geometry':record(rdir/'canonical_geometry.npz'),'target':record(target),'condition':record(cp),'source_review':record(review_by[obj]['review_path']),'target_axes':'proper depthX,lengthY,upZ directly encoded, no legacy permutation'})
recipe=root/'configs/gen/ss_flow_lora_64_canon_axis10_dinov3_20k.json';base=root/'checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors'
m={'schema':SCHEMA,'project_root':str(root),'production_training_approved':False,'prepare_only':True,'target_frame':'proper_raw_depth_length_up','seed':42,'train_objects':list(TRAIN),'internal_holdout_objects':list(HOLDOUT),'objects':rows,'dependencies':{'encoded_manifest':record(enc),'render_batch_validation':record(batch/'batch_render_validation.json'),'source_batch_review':record(batch/'asset_review/review.json'),'recipe':record(recipe),'base':record(base)},'recipe_overrides':{'max_steps':50,'batch_size_per_gpu':4,'batch_split':2,'i_save':50,'i_log':10,'i_print':10,'i_sample':51,'snapshot_batch_size':4,'prefetch_data':False},'heldout_external':['data_test/001.png','data_test/002.png','data_test/003.png','data_test/004.png'],'training_condition_policy':'pipeline preprocess alpha>204,crop_scale1,multiplyalpha RGB quantize then DINO512 LANCZOS; tensors exactly cachedRGB512','approval_required':'independent exact6encoded,targetvalidationpassed,split4/2,50step,diagnostic_only','nose_policy':'preserve reviewed asset selected side; no nose-flip/mirror; only samecamera view000','disk_budget':'require32GiB free conservative; one50step live+EMA group approx5.3GB','future_resume':'not implemented; phase50 exits; no automatic100 continuation'}
path=out/'manifest.json';path.write_text(json.dumps(m,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha(path),'preparation_complete':True,'training_ready':False},indent=2));validate(path,diagnostic=True);print(json.dumps({'manifest_sha256':sha(path),'train':list(TRAIN),'internal_holdout':list(HOLDOUT),'training_started':False},indent=2))
