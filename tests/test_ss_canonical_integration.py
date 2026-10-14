import unittest,tempfile,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
from data_toolkit.build_ss_canonical_contract import stage

class Integration(unittest.TestCase):
 def fixture(self,td):
  r=Path(td)/'data';r.mkdir();native=r/'renders_cond'/'native';native.mkdir(parents=True);t=r/'renders_cond_t';t.mkdir();(t/'native').symlink_to(native,target_is_directory=True);(t/'derived').mkdir()
  rows=[{'sha256':o,'cond_rendered':'true','aesthetic_score':'5'} for o in ['native','derived']]
  with open(r/'metadata.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  angles={}
  good=np.array([[i,j,k] for i in range(50) for j in range(3) for k in range(4)],np.int16)
  for o in ['native','derived']:
   frames=[];angles[o]={};(r/'ss_latents_views_canon'/o).mkdir(parents=True);(r/'ss_view_voxels_canon_v4'/o).mkdir(parents=True)
   for i in range(24):
    name=f'{i:03d}.png';frames.append({'file_path':name,'yaw_deg':i*15});angles[o][name]=0 if i in [0,12] else 90
    rgba=np.zeros((32,32,4),np.uint8);rgba[12:20,2:30,:]=255;Image.fromarray(rgba).save(t/o/name)
    np.savez(r/'ss_latents_views_canon'/o/f'{i:03d}.npz',mean=np.full((8,16,16,16),i,np.float32))
    np.savez(r/'ss_view_voxels_canon_v4'/o/f'{i:03d}.npz',coords=good if i in [0,12] else good[:,[1,0,2]])
   (t/o/'transforms.json').write_text(json.dumps({'frames':frames}))
  (t/'_view_offaxis.json').write_text(json.dumps(angles));return r
 def test_only_native_and_accepted_target(self):
  with tempfile.TemporaryDirectory() as td:
   r=self.fixture(td);out=Path(td)/'contract';stage(r,out,10);m=json.loads((out/'manifest.json').read_text());by={x['object']:x for x in m['objects']}
   self.assertTrue(by['native']['admitted']);self.assertEqual(by['native']['candidate_target_view'],0);self.assertEqual(by['native']['input_views'],[0,12]);self.assertFalse(by['derived']['admitted']);self.assertFalse((out/'canonical/derived.npz').exists());self.assertTrue((out/'COMPLETE.json').exists())
 def test_corrupt_target_fails_without_complete(self):
  with tempfile.TemporaryDirectory() as td:
   r=self.fixture(td);np.savez(r/'ss_latents_views_canon/native/000.npz',mean=np.full((8,16,16,16),np.nan,np.float32));out=Path(td)/'contract'
   with self.assertRaises(RuntimeError):stage(r,out,10)
   self.assertFalse((out/'COMPLETE.json').exists())
 def test_negative_cache_rejected_not_approved(self):
  with tempfile.TemporaryDirectory() as td:
   r=self.fixture(td);p=r/'renders_cond_t/_view_offaxis.json';a=json.loads(p.read_text());a['native']['000.png']=-1;p.write_text(json.dumps(a));out=Path(td)/'contract'
   with self.assertRaises(RuntimeError):stage(r,out,10)
   self.assertFalse((out/'COMPLETE.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
