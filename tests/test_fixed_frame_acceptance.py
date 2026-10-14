"""固定帧（fixed-frame）验收脚本 scripts/utils/evaluate_fixed_frame 的 CPU 单元测试。

- 覆盖点：几何朝向判据（最长 AABB 轴 + PCA 主轴与 Z 的偏角）不能被“长 Z”掩盖、
  场景世界变换与重复实例的合并、空/退化/非有限网格必须拒绝、缺失产物记为 fail 而非跳过、
  几何“未变差”在无人审时必须 unverified、raw_fast_glb 与 --ss_canonicalize 等快速/改轴路径一律 fail、
  运行证据（run manifest）的哈希与标志校验（含 v2 schema）、曲面 PCA 对三角化/重网格化的不变性、
  报告保留缺失记录且拒绝覆盖已有报告目录。
- 运行：python -m pytest tests/test_fixed_frame_acceptance.py（纯 CPU，用临时目录内自造 GLB/JSON）。
- 依赖：numpy、trimesh、scripts.utils.evaluate_fixed_frame（只读，不修改）。
"""
import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scripts.utils.evaluate_fixed_frame import world_mesh,mesh_metrics,evaluate_record,run

class FixedFrameTests(unittest.TestCase):
    # 测试用物体：x 细、y 中、z 最长的方盒，所以“最长轴 = z”且主轴应与 Z 对齐。
    def box(self):return trimesh.creation.box(extents=[.12,.2,1])
    # 契约：轴对齐盒子的最长轴为 z、朝向判定 pass、主轴与 Z 偏角 <2°，且度量过程不得改动输入网格。
    def test_aligned_box_orientation_and_no_mutation(self):
        mesh=self.box();before=mesh.vertices.copy();r=mesh_metrics(mesh)
        self.assertEqual(r['longest_aabb_axis'],'z');self.assertEqual(r['orientation_reference_status'],'pass')
        self.assertLess(r['pca_major_to_z_offangle_degrees'],2)
        np.testing.assert_array_equal(mesh.vertices,before)
    # 契约：把盒子绕 X 倾斜 25° 后，最长轴仍是 z，但 PCA 偏角要如实反映 25° 并判 fail（不能只看 AABB）。
    def test_tilt_not_hidden_by_long_z(self):
        mesh=self.box();mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(25),[1,0,0]));r=mesh_metrics(mesh)
        self.assertEqual(r['longest_aabb_axis'],'z');self.assertEqual(r['orientation_reference_status'],'fail')
        self.assertAlmostEqual(r['pca_major_to_z_offangle_degrees'],25,delta=2)
    # 契约：scene→mesh 必须应用节点世界变换，同一几何被引用两次（重复实例）要各展开一份。
    def test_scene_world_transform_and_duplicate_instances(self):
        mesh=self.box();scene=trimesh.Scene();m=trimesh.transformations.translation_matrix([2,3,4]);scene.add_geometry(mesh,geom_name='box',node_name='instance1',transform=m)
        scene.graph.update(frame_to='instance2',matrix=trimesh.transformations.translation_matrix([5,3,4]),geometry='box')
        result=world_mesh(scene);self.assertEqual(len(result.faces),2*len(mesh.faces))
        np.testing.assert_allclose(result.bounds,[[1.94,2.9,3.5],[5.06,3.1,4.5]])
        np.testing.assert_allclose(mesh.bounds,[[-.06,-.1,-.5],[.06,.1,.5]])
    # 契约：空场景、完全共线（零面积）网格、空网格都要抛 ValueError，不产出度量。
    def test_empty_and_degenerate_rejected(self):
        with self.assertRaises(ValueError):world_mesh(trimesh.Scene())
        with self.assertRaises(ValueError):mesh_metrics(trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[2,0,0]],faces=[[0,1,2]],process=False))
        with self.assertRaises(ValueError):mesh_metrics(trimesh.Trimesh())
    # 契约：顶点含 NaN 时度量必须报错（不允许把非有限值算进 PCA）。
    def test_nonfinite_rejected(self):
        mesh=self.box();mesh.vertices[0,0]=np.nan
        with self.assertRaises(ValueError):mesh_metrics(mesh)
    # 契约：合法面之外再插入一个退化面，退化面计数必须为 1（计数而非直接拒绝）。
    def test_degenerate_count_with_valid_faces(self):
        mesh=self.box();mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=np.vstack([mesh.faces,[0,0,0]]),process=False)
        self.assertEqual(mesh_metrics(mesh)['degenerate_faces'],1)
    # 契约：产物缺失记为 artifact_status=missing、验收 fail（绝不当成“跳过”而放过）。
    def test_missing_is_fail_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            r=evaluate_record({'id':'003','glb':str(Path(td)/'missing.glb'),'export_contract':'formal_standard_glb'})
            self.assertEqual(r['artifact_status'],'missing');self.assertEqual(r['acceptance_status'],'fail')
    # 契约：没有人工复核时几何“未变差”只能是 unverified；review 必须带 reviewer/evidence/rationale，
 # 只有 status=pass 的记录仍为 unverified，且 provenance 记为 provenance_unverified。
    def test_geometry_unverified_without_manual_review(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'a.glb';self.box().export(path)
            record={'id':'001','glb':str(path),'export_contract':'formal_standard_glb'};r=evaluate_record(record)
            self.assertEqual(r['orientation_reference_status'],'pass');self.assertEqual(r['geometry_not_worse_status'],'unverified');self.assertEqual(r['acceptance_status'],'unverified')
            with self.assertRaises(ValueError):evaluate_record(record,review={'status':'pass'})
            reviewed=evaluate_record(record,review={'status':'pass','reviewer':'test-fixture','evidence':'synthetic reference','rationale':'test only'})
            self.assertEqual(reviewed['acceptance_status'],'unverified')
            self.assertEqual(reviewed['provenance_status'],'provenance_unverified')
    # 契约：raw_fast_glb 直接 fail；命令里出现 --ss_canonicalize（改轴）也一律 fail。
    def test_fast_export_and_axis_correction_rejected(self):
        r=evaluate_record({'glb':'unused.glb','export_contract':'raw_fast_glb'});self.assertEqual(r['artifact_status'],'fail')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.box().export(root/'a.glb');(root/'command.txt').write_text('python infer.py --ss_canonicalize')
            r=evaluate_record({'id':'x','glb':str(root/'a.glb'),'export_contract':'formal_standard_glb','command_file':str(root/'command.txt')})
            self.assertEqual(r['acceptance_status'],'fail')
    # 契约：必须有完整运行证据（模型/输入/输出哈希 + 标志全 False + strict_validated）才可能 pass；
 # 把输出哈希改错后 provenance 变为 provenance_unverified 且验收不再 pass。
    def test_verified_manifest_required_and_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);glb=root/'a.glb';self.box().export(glb);inp=root/'input.png';inp.write_bytes(b'synthetic fixture')
            models=[]
            for role in ['sparse_structure','shape_1024','vae_decoder']:
                p=root/(role+'.pt');p.write_bytes(role.encode());models.append({'role':role,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            mp=root/'run.json';evidence={'schema':'trellis.fixed-frame-run-evidence.v1','status':'success','export_contract':'formal_standard_glb','output':{'path':str(glb),'sha256':hashlib.sha256(glb.read_bytes()).hexdigest()},'run':{'seed':42,'pipeline_type':'1024','flags':{k:False for k in ['ss_canonicalize','ss_swap_yz','fast_glb','geometry_only']}},'load_validation':'strict_validated','loaded_models':models,'input':{'path':str(inp),'sha256':hashlib.sha256(inp.read_bytes()).hexdigest()}}
            mp.write_text(json.dumps(evidence));record={'id':'fixture','glb':str(glb),'input':str(inp),'export_contract':'formal_standard_glb','run_manifest_path':str(mp)}
            review={'status':'pass','reviewer':'fixture','evidence':'synthetic only','rationale':'test'}
            result=evaluate_record(record,review=review);self.assertEqual(result['provenance_status'],'verified');self.assertEqual(result['acceptance_status'],'pass')
            evidence['output']['sha256']='wrong';mp.write_text(json.dumps(evidence));result=evaluate_record(record,review=review)
            self.assertEqual(result['provenance_status'],'provenance_unverified');self.assertNotEqual(result['acceptance_status'],'pass')
    # 契约：v2 运行清单（trellis.inference-run.v2）的哈希/标志校验——完整证据可 pass；
 # 输入哈希缺失或 fast_glb=True 都不得 pass。
    def test_native_v2_manifest_hashes_flags_and_manual_review(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);glb=root/'a.glb';self.box().export(glb);inp=root/'input.png';inp.write_bytes(b'fixture');mp=root/'run.json'
            def rec(p):return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            flags={k:False for k in ['ss_canonicalize','ss_swap_yz','fast_glb','geometry_only']};flags.update(seed=42,pipeline_type='1024')
            evidence={'schema':'trellis.inference-run.v2','status':'success','flags':flags,'seed':42,'pipeline_type':'1024','actual_coordinate_flags':{'ss_canonicalize':False,'ss_swap_yz':False},'acceptance_flags_eligible':True,'export':{'mode':'formal_remesh_textured','formal_acceptance_eligible':True},'actual_base_sources':{k:{'config':rec(inp),'weights':rec(inp)} for k in ['sparse_structure_flow_model','sparse_structure_decoder','shape_slat_flow_model_1024','shape_slat_decoder']},'resolved_weights':{},'loaded_components':[],'images':[{'input':str(inp),'input_artifact':rec(inp),'status':'success','outputs':[rec(glb)]}]}
            mp.write_text(json.dumps(evidence));record={'id':'fixture','glb':str(glb),'input':str(inp),'export_contract':'formal_standard_glb','run_manifest_path':str(mp)};review={'status':'pass','reviewer':'fixture','evidence':'synthetic','rationale':'test'}
            self.assertEqual(evaluate_record(record,review=review)['acceptance_status'],'pass')
            evidence['images'][0]['input_artifact'].pop('sha256');mp.write_text(json.dumps(evidence));self.assertNotEqual(evaluate_record(record,review=review)['acceptance_status'],'pass')
            evidence['images'][0]['input_artifact']=rec(inp);evidence['flags']['fast_glb']=True;mp.write_text(json.dumps(evidence));self.assertNotEqual(evaluate_record(record,review=review)['acceptance_status'],'pass')
    # 契约：曲面 PCA（面积加权）对细分三角化不变——单个三角形的均值和协方差必须逐位（1e-13）一致。
    def test_surface_moments_single_triangle_split_invariant(self):
        vertices=np.array([[0.,0.,0.],[.8,.1,0.],[.1,.3,1.4]])
        mesh=trimesh.Trimesh(vertices=vertices,faces=[[0,1,2]],process=False)
        subdivided=mesh.subdivide().subdivide()
        a=mesh_metrics(mesh)['surface_pca'];b=mesh_metrics(subdivided)['surface_pca']
        np.testing.assert_allclose(a['mean'],b['mean'],atol=1e-13)
        np.testing.assert_allclose(a['covariance'],b['covariance'],atol=1e-13)
        np.testing.assert_allclose(a['eigenvalues_ascending'],b['eigenvalues_ascending'],atol=1e-13)
    # 契约：XY 平面上 2x4 矩形的解析矩（均值 0、协方差 diag(1/3,0,4/3)）必须被精确复现，且与细分无关。
    def test_surface_rectangle_retessellation_analytic_moments(self):
        v=np.array([[-1,0,-2],[1,0,-2],[1,0,2],[-1,0,2]],float)
        mesh=trimesh.Trimesh(vertices=v,faces=[[0,1,2],[0,2,3]],process=False)
        a=mesh_metrics(mesh)['surface_pca'];b=mesh_metrics(mesh.subdivide().subdivide())['surface_pca']
        np.testing.assert_allclose(a['mean'],[0,0,0],atol=1e-13)
        np.testing.assert_allclose(a['covariance'],np.diag([1/3,0,4/3]),atol=1e-13)
        np.testing.assert_allclose(a['covariance'],b['covariance'],atol=1e-13)
    # 契约：float32 精度下旋转 17° 的盒子，协方差与细分后一致（容差 2e-6），主轴偏角精确为 17°。
    def test_surface_rotated_float32_box_tessellation_invariant(self):
        mesh=self.box();mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(17),[1,0,0]));mesh.vertices=np.asarray(mesh.vertices,dtype=np.float32)
        fine=mesh.subdivide().subdivide();fine.vertices=np.asarray(fine.vertices,dtype=np.float32)
        a=mesh_metrics(mesh)['surface_pca'];b=mesh_metrics(fine)['surface_pca']
        np.testing.assert_allclose(a['covariance'],b['covariance'],rtol=2e-6,atol=2e-9)
        self.assertAlmostEqual(a['major_to_z_offangle_degrees'],17,delta=1e-5)
        self.assertAlmostEqual(a['major_to_z_offangle_degrees'],b['major_to_z_offangle_degrees'],delta=1e-5)
    # 契约：报告必须为每条记录留结果（含第 4 条缺失记录），整体验收为 fail；同一输出目录拒绝覆盖。
    def test_report_keeps_missing_fourth_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.box().export(root/'a.glb');records=[{'id':str(i),'glb':str(root/('a.glb' if i<3 else 'missing.glb')),'export_contract':'formal_standard_glb'} for i in range(4)]
            report=run({'records':records},root/'report');self.assertEqual(len(report['results']),4);self.assertEqual(report['all_records_acceptance'],'fail')
            with self.assertRaises(FileExistsError):run({'records':records},root/'report')
if __name__=='__main__':unittest.main(verbosity=2)
