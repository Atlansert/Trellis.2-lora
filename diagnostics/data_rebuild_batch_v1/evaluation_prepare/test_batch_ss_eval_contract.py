import unittest,tempfile,json
from pathlib import Path
import numpy as np
from batch_ss_eval_contract import occupancy,metrics,summary,fresh_output,artifact,sha,validate_inputs
class EvaluationTests(unittest.TestCase):
 def test_exact_iou_no_alignment(self):
  c=np.array([[1,20,2],[1,21,2]]);self.assertEqual(metrics(c,c)['target']['iou'],1);moved=c+np.array([10,0,0]);self.assertEqual(metrics(moved,c)['target']['iou'],0)
 def test_known_up_depth_coordinates_unchanged(self):
  c=np.array([[2,8,12],[3,50,13]]);m=metrics(c,c);self.assertEqual(m['spans_depth_length_up'],[2,43,2]);self.assertEqual(m['longest_index'],1)
 def test_invalid_coordinates_fail(self):
  for c in [np.array([[64,0,0]]),np.array([[np.nan,0,0]]),np.array([[.5,0,0]]),np.ones((2,2)),np.array([[1,2,3,4]])]:
   with self.assertRaises(ValueError):occupancy(c)
 def test_empty_explicit(self):self.assertEqual(metrics(np.zeros((0,3)))['axis_status'],'empty')
 def test_external_without_gt_has_no_iou(self):
  r=metrics([[1,2,3],[1,4,3]]);self.assertNotIn('target',r);self.assertIn('no_gt',r);self.assertEqual(r['fidelity_status'],'unverified')
 def test_scene_batch_column_must_zero(self):
  self.assertEqual(metrics([[0,1,2,3],[0,1,4,3]])['count'],2)
  with self.assertRaises(ValueError):metrics([[1,1,2,3]])
 def test_no_output_overwrite(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(FileExistsError):fresh_output(t)
 def test_artifact_tamper_and_traversal(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x';p.write_text('hello');r={'path':str(p),'sha256':sha(p)};artifact(r);p.write_text('changed')
   with self.assertRaises(ValueError):artifact(r)
   with self.assertRaises(ValueError):artifact({'path':'../x','sha256':'0'*64})
 def test_summaries_use_all_objects_min_and_mean(self):
  cases=[]
  for split,values in [('train',[(.2,.4),(.8,.6)]),('internal_holdout',[(.4,.3)])]:
   for base,live in values:
    for tag,value in [('base',base),('live',live)]:cases.append({'split':split,'tag':tag,'target':{'iou':value}})
  r=summary(cases);self.assertAlmostEqual(r[0]['mean_change'],0);self.assertEqual(r[0]['live_min_iou'],.4);self.assertAlmostEqual(r[1]['mean_change'],-.1)
 def test_incomplete_split_not_false_pass(self):
  with self.assertRaises(ValueError):summary([{'split':'train','tag':'base','target':{'iou':1}}])
if __name__=='__main__':unittest.main(verbosity=2)
