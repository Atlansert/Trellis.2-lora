"""规范契约“加载侧”的 CPU 单元测试：manifest/COMPLETE/批准文件与数据集封装的门禁。

- 覆盖点：暂存（staged）契约默认被拒（必须显式 diagnostic 打开）；manifest 被改一个字节即校验和失败；
  源图被改即 'source changed'；缺 COMPLETE.json 即 FileNotFoundError；只写 approved=True 而无证据的
  TRAINING_APPROVAL.json 无效；manifest 的 root 与实际 root 不一致被拒；数据集封装在初始化后
  目标文件被改、过滤条件（min_aesthetic_score/num_views）变化都必须报错。
- 运行：python -m pytest tests/test_ss_contract_loading.py（纯 CPU；会真的实例化数据集但只读小样本）。
- 依赖：test_ss_canonical_integration 的 fixture、data_toolkit.build_ss_canonical_contract.stage、
  trellis2.datasets 下的 canonical_contract 与 sparse_structure_latent_views。
"""
import unittest,tempfile,json,hashlib
from pathlib import Path
import test_ss_canonical_integration as fixtures
from data_toolkit.build_ss_canonical_contract import stage
from trellis2.datasets.canonical_contract import load_canonical_contract,file_sha256
from trellis2.datasets.sparse_structure_latent_views import ImageConditionedSparseStructureLatentViews as Dataset

class LoadingTests(unittest.TestCase):
 # 复用集成测试的建数据逻辑：先 stage 出一个规范契约目录。
 def fixture(self,td):
  r=fixtures.Integration().fixture(td);out=Path(td)/'contract';stage(r,out,10);return r,out
 # 契约：未获批准（staged）的契约默认不可用于训练，必须报 'not approved'。
 def test_staged_denied_by_default(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td)
   with self.assertRaisesRegex(ValueError,'not approved'):load_canonical_contract(c/'manifest.json',c/'canonical')
 # 契约：显式 diagnostic 打开后只返回 native 对象，且 manifest 里记录的 root 必须等于真实 root。
 def test_explicit_diagnostic_open(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);rr,o=load_canonical_contract(c/'manifest.json',c/'canonical',True);self.assertEqual(set(o),{'native'});self.assertEqual(rr,r.resolve())
 # 契约：manifest.json 多一个空格就改变哈希，必须报 'checksum' 失败。
 def test_manifest_tamper_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);p=c/'manifest.json';p.write_text(p.read_text()+' ')
   with self.assertRaisesRegex(ValueError,'checksum'):load_canonical_contract(p,c/'canonical',True)
 # 契约：输入图被改动后，manifest 里记录的源图哈希对不上，必须报 'source changed'。
 def test_source_image_tamper_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);p=r/'renders_cond_t/native/000.png';p.write_bytes(p.read_bytes()+b'x')
   with self.assertRaisesRegex(ValueError,'source changed'):load_canonical_contract(c/'manifest.json',c/'canonical',True)
 # 契约：缺少 COMPLETE.json（说明暂存未跑完）时加载必须报 FileNotFoundError。
 def test_missing_complete_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);(c/'COMPLETE.json').unlink()
   with self.assertRaises(FileNotFoundError):load_canonical_contract(c/'manifest.json',c/'canonical',True)
 # 契约：只写 approved=True（无配套证据）的批准文件不算数，必须报 'approval' 相关错误。
 def test_approval_boolean_without_evidence_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);(c/'TRAINING_APPROVAL.json').write_text(json.dumps({'approved':True,'manifest_sha256':file_sha256(c/'manifest.json')}))
   with self.assertRaisesRegex(ValueError,'approval'):load_canonical_contract(c/'manifest.json',c/'canonical')
 # 契约：把契约目录当 root 传进来（root 不匹配）必须被拒绝。
 def test_root_mismatch_denied(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td)
   with self.assertRaisesRegex(ValueError,'root'):load_canonical_contract(c/'manifest.json',r,True)
 # 契约：数据集侧的门禁——未批准时构造失败；准许 staged 后可读样本（x_0 形状 (8,16,16,16)、
 # cond 为 3×512×512），初始化后目标 npz 被改即报 'after initialization'，改变过滤条件
 # （min_aesthetic_score / num_views）必须报错，防止训练集与契约不一致。
 def test_dataset_contract_views_and_staged_guard(self):
  with tempfile.TemporaryDirectory() as td:
   r,c=self.fixture(td);roots={'test':{'base':str(r),'render_cond':str(r/'renders_cond_t'),'ss_view_latent':str(r/'ss_latents_views_canon')}};kwargs=dict(min_aesthetic_score=4.5,num_views=24,latent_frame='canonical',canonical_latent_root=str(c/'canonical'),canonical_manifest=str(c/'manifest.json'),max_view_offaxis=10)
   with self.assertRaisesRegex(ValueError,'not approved'):Dataset(roots,**kwargs)
   d=Dataset(roots,allow_staged_contract=True,**kwargs);self.assertEqual(len(d),1);self.assertEqual(d.instances[0][3],[0,12])
   sample=d[0];self.assertEqual(tuple(sample['x_0'].shape),(8,16,16,16));self.assertEqual(tuple(sample['cond'].shape),(3,512,512))
   target=c/'canonical/native.npz';original=target.read_bytes();target.write_bytes(original+b'x')
   with self.assertRaisesRegex(ValueError,'after initialization'):d[0]
   target.write_bytes(original)
   kwargs['min_aesthetic_score']=6
   with self.assertRaisesRegex(ValueError,'object set'):Dataset(roots,allow_staged_contract=True,**kwargs)
   kwargs['min_aesthetic_score']=4.5
   kwargs['num_views']=1
   with self.assertRaisesRegex(ValueError,'filters change'):Dataset(roots,allow_staged_contract=True,**kwargs)
if __name__=='__main__':unittest.main(verbosity=2)
