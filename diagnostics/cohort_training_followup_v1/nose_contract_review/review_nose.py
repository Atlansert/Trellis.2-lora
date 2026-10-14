import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image,ImageDraw
root=Path('/data/lzx/TRELLIS.2');b=root/'diagnostics/data_rebuild_batch_v1';out=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((b/'ss50_evaluation_v1/metrics.json').read_text());prior=json.loads((b/'heading_diagnostic_not_acceptance.json').read_text());pm={(x['input'],x['tag']):x for x in prior['rows']}
reviews=json.loads((b/'asset_review/review.json').read_text());records=[];canvas=Image.new('RGB',(1800,6*260),(25,28,32));draw=ImageDraw.Draw(canvas)
def taper(c):
 lo,hi=c[:,1].min(),c[:,1].max();d=.08*(hi-lo);ends=[]
 for sign,mask in [(-1,c[:,1]<=lo+d),(1,c[:,1]>=hi-d)]:
  z=c[mask];span=np.ptp(z,axis=0);ends.append({'sign':sign,'n':len(z),'depth_span':float(span[0]),'up_span':float(span[2]),'bbox_area_proxy':float(span[0]*span[2])})
 values=[x['bbox_area_proxy'] for x in ends];candidate=ends[int(np.argmin(values))]['sign'];ratio=max(values)/max(min(values),1e-12)
 return {'ends':ends,'candidate_nose_rawY_sign':candidate,'taper_area_ratio':ratio,'algorithm':'smaller depth*height bbox in each end8% length region; candidate not measured semantic keypoint'}
def projection(c):
 c=np.asarray(c);c=c[:,1:] if c.shape[1]==4 else c;im=Image.new('RGB',(300,210),(25,28,32));d=ImageDraw.Draw(im)
 # Coordinates already grid64; same camera+X, right+Y, up+Z.
 for q in c:d.rectangle((int(q[1]*4.4)+8,185-int(q[2]*4.4),int(q[1]*4.4)+11,188-int(q[2]*4.4)),fill=(85,190,210))
 return im
for i,e in enumerate(reviews['selected']):
 oid=e['object_id'];rd=b/'renders'/f'{oid}_cpu_v1';geo=rd/'canonical_geometry.npz';gtfile=b/'encoded_targets_v1'/oid/'ss_support64.npz'
 with np.load(geo) as z:v=z['vertices'];f=z['faces']
 with np.load(gtfile) as z:gt=z['coords'].astype(float)
 rec={'object_id':oid,'selected_source_yaw':e['selected_yaw_degrees'],'source':e['source'],'geometry_sha256':sha(geo),'GT_support_sha256':sha(gtfile),'geometry_taper':taper(v),'GT_support_taper':taper(gt),'predictions':[],'images':[]}
 for j,n in enumerate(['000.png','004.png']):
  ip=rd/n;im=Image.open(ip).convert('RGBA');rgb=Image.new('RGB',im.size,'black');rgb.paste(im,mask=im.getchannel('A'));rgb=rgb.resize((300,300),Image.Resampling.LANCZOS);canvas.paste(rgb.crop((0,40,300,250)),(j*300,i*260+30));rec['images'].append({'view':n,'sha256':sha(ip),'matrix_camera_to_world':json.loads((rd/'manifest.json').read_text())['views'][int(n[:3])]['matrix_world_camera_to_world']})
 canvas.paste(projection(gt),(600,i*260+30))
 for j,tag in enumerate(['base','live']):
  c=next(x for x in m['cases'] if x['tag']==tag and x['input']==oid);path=Path(c['coords_artifact']['path'])
  with np.load(path) as z:coords=z['coords'][:,1:].astype(float)
  t=taper(coords);h=pm[oid,tag];rec['predictions'].append({'tag':tag,'coords_sha256':sha(path),'taper':t,'direct_iou':h['direct_iou'],'yaw180_diagnostic_iou':h['yaw180_diagnostic_iou'],'yaw180_minus_direct':h['yaw180_diagnostic_iou']-h['direct_iou'],'yaw180_is_hypothesis_not_applied':True});canvas.paste(projection(coords),(900+j*300,i*260+30))
  draw.text((900+j*300,i*260+13),tag+' nose candidate '+str(t['candidate_nose_rawY_sign'])+'Y',fill='white')
 draw.text((5,i*260),oid+' source0 / source180 / GTSS / base / live50; screen right raw+Y, up+Z',fill='white');draw.text((600,i*260+13),'GT candidate '+str(rec['GT_support_taper']['candidate_nose_rawY_sign'])+'Y',fill='white')
 text='GT mesh taper '+str(rec['geometry_taper']['candidate_nose_rawY_sign'])+'Y ratio '+format(rec['geometry_taper']['taper_area_ratio'],'.2f')
 draw.text((1500,i*260+50),text,fill='white');draw.text((1500,i*260+75),'Program candidates, not truth',fill='white');records.append(rec)
canvas.save(out/'six_nose_contact.png');r={'schema':'trellis.six-nose-review.v1','camera':'0 +rawX looks-X/up+Z; screenright+rawY=-formalZ; oppositecamera180 reverses screen direction without mirroring mesh','acceptance':'direct unaligned IoU ONLY; yaw180hypothesis never applied','records':records,'training_approved':False,'source_geometry_modified':False,'inputs':{'SS_metrics_sha256':sha(b/'ss50_evaluation_v1/metrics.json'),'prior_heading_sha256':sha(b/'heading_diagnostic_not_acceptance.json'),'asset_review_sha256':sha(b/'asset_review/review.json')}};(out/'review.json').write_text(json.dumps(r,indent=2));print([(x['object_id'],x['geometry_taper']['candidate_nose_rawY_sign'],x['GT_support_taper']['candidate_nose_rawY_sign'],[(y['tag'],y['taper']['candidate_nose_rawY_sign'],y['yaw180_minus_direct']) for y in x['predictions']]) for x in records])
