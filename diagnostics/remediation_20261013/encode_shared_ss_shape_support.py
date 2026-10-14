import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,time
from pathlib import Path
root=Path('/data/lzx/TRELLIS.2');sys.path.insert(0,str(root));os.chdir(root)
import torch,numpy as np
from trellis2 import models
torch.set_num_threads(8)
base=root/'diagnostics/remediation_20261013/common_shape_pilot';out=root/'diagnostics/remediation_20261013/shared_ss_shape_encoded';out.mkdir(exist_ok=False)
enc_prefix='/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16';dec_prefix='/home/test/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16'
encoder=models.from_pretrained(enc_prefix);decoder=models.from_pretrained(dec_prefix)
for m in [encoder,decoder]:m.convert_to_fp32();m.float().cpu().eval()
records=[]
for obj in ['0086bbab37559442','0fab4af3f9a73bdd']:
 p=base/obj/'ss_support64_from_shape_encoder.npz';c=np.load(p,allow_pickle=False)['coords'].astype(int);volume=torch.zeros((1,1,64,64,64));volume[0,0,c[:,0],c[:,1],c[:,2]]=1
 with torch.inference_mode():
  z=encoder(volume,sample_posterior=False);logits=decoder(z);pred=logits>0
 assert z.shape==(1,8,16,16,16) and torch.isfinite(z).all() and torch.isfinite(logits).all();gt=volume>0;iou=(pred & gt).sum().item()/(pred | gt).sum().item()
 dst=out/f'{obj}.npz';np.savez_compressed(dst,mean=z[0].numpy());records.append({'object':obj,'source_shape_support_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tokens':len(c),'ss_roundtrip_iou':iou,'mean_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'finite_fp32':True})
result={'schema':'trellis.shared-ss-shape-encoded.v1','training_approved':False,'encoder':enc_prefix,'decoder':dec_prefix,'same_axis_no_legacy_permutation':True,'records':records};(out/'manifest.json').write_text(json.dumps(result,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),'training_approved':False},indent=2));print(json.dumps(result,indent=2))
