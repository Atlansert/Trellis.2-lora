#!/usr/bin/env python3
"""固定帧（fixed world frame）验收诊断：纯 CPU，只做诊断，不是自动的几何保真闸门。

用途：对已有推理产物按"固定世界坐标 + 标准 GLB 契约"复核朝向/几何，并校验 run_manifest 溯源链。
关键输入：--root（内置历史清单）或 --manifest（外部清单）+ --manual-reviews；关键输出：<output>/report.json 与 input_manifest.json。
要点：只接受 formal 标准 GLB（raw fast_glb/geometry_only 帧契约不同，直接判 fail）；长度轴应为 Z；PCA 用表面面积积分协方差；10° 只是诊断参考阈值，不是生产验收规则。
三态结论：pass / fail / unverified —— 溯源不可证或 PCA 轴歧义时保持 unverified，绝不靠几何好看"升级"为 pass。
安全边界：只读 GLB/日志/manifest，只写 --output 目录（目录已存在则报错）；不做网格形变、不做最佳视角搜索、不加载 checkpoint 张量。

Original (English):
CPU fixed-world-frame diagnostics, never an automatic geometry fidelity gate.

Standard GLB only: raw fast_glb/geometry_only exports require their own frame
contract and are rejected. No PCA alignment, best-view search, or mesh mutation.
Orientation threshold is a diagnostic reference, not a replacement user gate.
"""
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np
import trimesh

def sha256(path):
    """分块计算文件 sha256（不整文件读进内存）。"""
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def world_mesh(scene):
    """Apply every scene instance's full world transform, including duplicates.
    把场景里每个实例的世界变换都应用后合并成一个网格（重复实例也算），得到固定世界帧下的真实几何。
    """
    if isinstance(scene,trimesh.Trimesh):return scene.copy()
    meshes=[]
    for node in scene.graph.nodes_geometry:
        matrix,name=scene.graph[node]
        geometry=scene.geometry[name]
        if not isinstance(geometry,trimesh.Trimesh):continue
        mesh=geometry.copy();mesh.apply_transform(matrix);meshes.append(mesh)
    if not meshes:raise ValueError('empty scene: no triangle mesh instances')
    return trimesh.util.concatenate(meshes)

def mesh_metrics(mesh,reference_angle=10):
    """固定帧几何度量：AABB 跨度 + 表面面积积分协方差 PCA 主方向(判断长度轴是否落在 Z)。
    退化面按相对面积容差剔除；PCA 主/次特征值比 <1.5 记为轴歧义；返回朝向参考状态 pass/fail。
    """
    v=np.asarray(mesh.vertices,dtype=np.float64);f=np.asarray(mesh.faces)
    if not len(v) or not len(f):raise ValueError('empty mesh vertices/faces')
    if not np.isfinite(v).all():raise ValueError('nonfinite vertices')
    if f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(v):raise ValueError('invalid faces')
    bounds=np.stack([v.min(0),v.max(0)]);extents=bounds[1]-bounds[0]
    triangles=v[f];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas=np.linalg.norm(cross,axis=1)*.5
    tolerance=max(float(extents.max())**2*1e-14,np.finfo(float).tiny)
    valid=np.isfinite(areas)&(areas>tolerance)
    degenerate=int((~valid).sum())
    if not valid.any():raise ValueError('all faces degenerate')
    # Area-weighted FACE CENTERS: unlike vertex PCA, UV seams and tessellation
    # vertex multiplicities do not directly bias the covariance. This remains
    # a surface sampling proxy, not volume PCA or a shape fidelity metric.
    centers=triangles[valid].mean(axis=1);weights=areas[valid];total=float(weights.sum())
    mean=np.sum(centers*weights[:,None],axis=0)/total
    delta=centers-mean;cov=(delta.T*weights)@delta/total
    eigenvalues,eigenvectors=np.linalg.eigh(cov);direction=eigenvectors[:,-1]
    if direction[2]<0:direction=-direction
    offangle=float(np.degrees(np.arccos(np.clip(abs(direction[2]),0,1))))
    gap=float(eigenvalues[-1]/max(eigenvalues[-2],np.finfo(float).tiny))
    # Exact uniform-triangle surface moments via law of total covariance.
    # Within-triangle covariance = sum_i (v_i-c)(v_i-c)^T / 12.
    # Keeping this term makes moments invariant to coplanar tessellation.
    # 面内协方差项(全协方差定律)：加上它，度量对共面三角化的疏密不敏感。
    surface_cov=cov.copy()
    for i in range(3):
        residual=triangles[valid,i,:]-centers
        surface_cov+=(residual.T*weights)@residual/(12*total)
    surface_eigenvalues,surface_eigenvectors=np.linalg.eigh(surface_cov)
    surface_direction=surface_eigenvectors[:,-1]
    if surface_direction[2]<0:surface_direction=-surface_direction
    surface_offangle=float(np.degrees(np.arccos(np.clip(abs(surface_direction[2]),0,1))))
    surface_gap=float(surface_eigenvalues[-1]/max(surface_eigenvalues[-2],np.finfo(float).tiny))
    # 朝向参考判据：最长 AABB 轴必须是 Z(下标 2)，且表面 PCA 主轴与 Z 的夹角不超过参考阈值。
    longest=int(np.argmax(extents));orientation='pass' if longest==2 and surface_offangle<=reference_angle else 'fail'
    return {'finite':True,'vertices':len(v),'faces':len(f),'degenerate_faces':degenerate,
            'degenerate_area_tolerance':tolerance,'aabb':bounds.tolist(),'extents_xyz':extents.tolist(),
            'longest_aabb_axis':'xyz'[longest],'surface_area':total,
            'facecenter_area_weighted_mean':mean.tolist(),'facecenter_area_weighted_covariance':cov.tolist(),
            'pca_eigenvalues_ascending':eigenvalues.tolist(),'pca_major_axis_world_xyz':direction.tolist(),
            'pca_major_to_z_offangle_degrees':offangle,'pca_major_to_second_eigenvalue_ratio':gap,
            'pca_axis_ambiguous':gap<1.5,
            'primary_orientation_method':'exact_uniform_triangle_surface_integrated_covariance',
            'surface_pca':{'mean':mean.tolist(),'covariance':surface_cov.tolist(),
                           'eigenvalues_ascending':surface_eigenvalues.tolist(),'major_axis_world_xyz':surface_direction.tolist(),
                           'major_to_z_offangle_degrees':surface_offangle,'major_to_second_eigenvalue_ratio':surface_gap,
                           'axis_ambiguous':surface_gap<1.5,'orientation_reference_status':orientation},
            'transverse_world_x_width':float(extents[0]),'transverse_world_y_height':float(extents[1]),
            'fixed_ZY_projection_bbox_width_height':[float(extents[2]),float(extents[1])],
            'fixed_ZX_projection_bbox_width_height':[float(extents[2]),float(extents[0])],
            'z_over_y':float(extents[2]/extents[1]) if extents[1]>0 else None,
            'z_over_x':float(extents[2]/extents[0]) if extents[0]>0 else None,
            'x_over_y':float(extents[0]/extents[1]) if extents[1]>0 else None,
            'orientation_reference_status':orientation,'reference_angle_degrees':reference_angle}

def log_provenance(path):
    """读取推理日志的溯源摘要：文件哈希、"加载 ...pt/safetensors" 行、是否出现 Traceback/错误标记。"""
    p=Path(path)
    if not p.is_file():return {'status':'missing','path':str(p)}
    text=p.read_text(encoding='utf-8',errors='replace')
    return {'status':'present','path':str(p),'sha256':sha256(p),
            'load_path_lines':[line.strip() for line in text.splitlines() if '加载' in line and ('.pt' in line or '.safetensors' in line)],
            'error_marker_present':bool(re.search(r'Traceback|错误:|Error:',text))}

class UnsafeRunContract(ValueError):
    """契约本身不安全（raw 导出/坐标修正等），判 fail；与"暂时证不了"的 unverified 区分开。"""
    pass

def hash_record_metadata(value):
    """校验一条哈希记录：path 存在、size_bytes 未变、sha256 为 64 位十六进制。checkpoint 载荷不重算。"""
    if not isinstance(value,dict) or not value.get('path') or not re.fullmatch(r'[0-9a-f]{64}',value.get('sha256','')):raise ValueError('incomplete hash record')
    path=Path(value['path'])
    if not path.is_file() or value.get('size_bytes')!=path.stat().st_size:raise ValueError('recorded source missing/size changed')

def verify_native_run_manifest(record,evidence,path,artifact_hash):
    """校验原生 trellis.inference-run.v2 manifest（v2 溯源）：
    seed/pipeline 必须为 42/1024，坐标与导出开关必须为安全值，base 与解析权重都要有完整哈希记录，
    且目标 GLB 在记录里唯一、其路径/哈希/大小与输入图片哈希都能对上现有文件。
    """
    if evidence.get('status')!='success':raise ValueError('inference run not successful')
    flags=evidence.get('options',evidence.get('flags',{}))
    seed=flags.get('seed',evidence.get('seed'));pipeline=flags.get('pipeline_type',evidence.get('pipeline_type'))
    if seed!=42 or str(pipeline)!='1024':raise ValueError('seed/pipeline acceptance contract mismatch')
    if evidence.get('seed',seed)!=seed or str(evidence.get('pipeline_type',pipeline))!=str(pipeline):raise ValueError('inconsistent seed/pipeline evidence')
    actual=evidence.get('actual_coordinate_flags',{})
    # 出现任何"raw 导出 / 坐标修正(ss_canonicalize, ss_swap_yz)"痕迹 -> 契约不安全，判 fail 而非 unverified。
    if any(actual.get(k) is True for k in ['ss_canonicalize','ss_swap_yz']) or any(flags.get(k) is True for k in ['ss_canonicalize','ss_swap_yz','fast_glb','geometry_only']) or evidence.get('export',{}).get('mode')=='raw_no_remesh':
        raise UnsafeRunContract('explicit raw export or coordinate correction not accepted')
    if any(actual.get(k) is not False for k in ['ss_canonicalize','ss_swap_yz']):raise ValueError('actual coordinate flags missing/unsafe')
    if any(flags.get(k) is not False for k in ['ss_canonicalize','ss_swap_yz','fast_glb','geometry_only']):raise ValueError('CLI coordinate/export flags missing/unsafe')
    export=evidence.get('export',{})
    if evidence.get('acceptance_flags_eligible') is not True or export.get('mode')!='formal_remesh_textured' or export.get('formal_acceptance_eligible') is not True:raise ValueError('formal acceptance export not verified')
    sources=evidence.get('actual_base_sources',{})
    required={'sparse_structure_flow_model','sparse_structure_decoder','shape_slat_flow_model_1024','shape_slat_decoder'}
    if not required.issubset(sources):raise ValueError('actual required base source records missing')
    for source in sources.values():
        hash_record_metadata(source.get('config'));hash_record_metadata(source.get('weights'))
    resolved=evidence.get('resolved_weights')
    if not isinstance(resolved,dict):raise ValueError('resolved weight records missing')
    for value in resolved.values():hash_record_metadata(value)
    loaded=evidence.get('loaded_components')
    if not isinstance(loaded,list):raise ValueError('loaded components missing')
    if any(key not in loaded for key in resolved):raise ValueError('resolved override not actually loaded')
    # 目标 GLB 必须在 manifest 中恰好对应一条 success 记录，避免多图/多产物串台。
    images=[x for x in evidence.get('images',[]) if x.get('status')=='success' and any(Path(o.get('path','')).resolve()==Path(record['glb']).resolve() for o in x.get('outputs',[]))]
    if len(images)!=1:raise ValueError('successful output image record not unique')
    item=images[0];matches=[x for x in item['outputs'] if Path(x['path']).resolve()==Path(record['glb']).resolve()]
    output=matches[0]
    if output.get('sha256')!=artifact_hash or output.get('size_bytes')!=Path(record['glb']).stat().st_size:raise ValueError('output GLB hash/size mismatch or missing')
    inp=Path(item['input'])
    if record.get('input') and inp.resolve()!=Path(record['input']).resolve():raise ValueError('input path mismatch')
    input_record=item.get('input_artifact');hash_record_metadata(input_record)
    if Path(input_record['path']).resolve()!=inp.resolve() or sha256(inp)!=input_record['sha256']:raise ValueError('input SHA256/path mismatch')
    return {'status':'verified','path':str(path),'sha256':sha256(path),'schema':evidence['schema'],'evidence_scope':'manifest provenance and live GLB/input hashes; checkpoint payload SHA records retained with path/size check, not rehashed or tensor-reloaded here','actual_base_sources':sources,'resolved_weights':resolved}

def verify_run_provenance(record,artifact_hash):
    """Never upgrade historical declarations using a geometry review alone.

    Explicit evidence schema until the inference writer's native schema is
    integrated: trellis.fixed-frame-run-evidence.v1. Unknown schemas fail closed.
    按 schema 分派溯源校验：trellis.inference-run.v2 走原生校验；历史 v1 evidence 需逐项自证；
    未知 schema 一律 fail-closed（结果保持 provenance_unverified/fail，绝不放行）。
    """
    path=record.get('run_manifest_path')
    if not path:return {'status':'provenance_unverified','reason':'historical declarations/log lines do not prove complete flags, seed and loaded weights'}
    p=Path(path)
    try:
        evidence=json.loads(p.read_text(encoding='utf-8'))
        if evidence.get('schema')=='trellis.inference-run.v2':
            return verify_native_run_manifest(record,evidence,p,artifact_hash)
        if evidence.get('schema')!='trellis.fixed-frame-run-evidence.v1':raise ValueError('unsupported run manifest schema')
        if evidence.get('status')!='success':raise ValueError('run not successful')
        if evidence.get('export_contract')!='formal_standard_glb':raise ValueError('not formal standard GLB')
        output=evidence['output']
        if Path(output['path']).resolve()!=Path(record['glb']).resolve() or output['sha256']!=artifact_hash:raise ValueError('GLB path/hash mismatch')
        run=evidence['run']
        if run.get('seed')!=42 or str(run.get('pipeline_type'))!='1024':raise ValueError('seed/pipeline contract mismatch')
        flags=run['flags']
        for key in ['ss_canonicalize','ss_swap_yz','fast_glb','geometry_only']:
            if flags.get(key) is not False:raise ValueError('missing/unsafe explicit flag '+key)
        if evidence.get('load_validation')!='strict_validated':raise ValueError('weight load not strictly validated')
        models=evidence['loaded_models']
        if not isinstance(models,list) or not models:raise ValueError('no loaded model evidence')
        required={'sparse_structure','shape_1024','vae_decoder'}
        if not required.issubset({x.get('role') for x in models}):raise ValueError('required model roles missing')
        for model in models:
            if not model.get('path') or not model.get('sha256'):raise ValueError('model path/hash evidence missing')
            if sha256(Path(model['path']))!=model['sha256']:raise ValueError('model file changed: '+model['role'])
        inp=evidence['input']
        if record.get('input') and Path(inp['path']).resolve()!=Path(record['input']).resolve():raise ValueError('input path mismatch')
        if sha256(Path(inp['path']))!=inp['sha256']:raise ValueError('input hash mismatch')
        return {'status':'verified','path':str(p),'sha256':sha256(p),'schema':evidence['schema']}
    except Exception as e:
        return {'status':'fail' if isinstance(e,UnsafeRunContract) else 'provenance_unverified','path':str(p),'reason':f'{type(e).__name__}: {e}'}

def evaluate_record(record,reference_angle=10,review=None):
    """评估单条记录并给出三态结论：证据缺失/不安全 -> fail；溯源未证实或 PCA 轴歧义 -> unverified；
    只有 artifact + orientation + geometry_not_worse + provenance 全部 pass 才汇总为 pass。
    """
    result={'record':record,'artifact_status':'missing','orientation_reference_status':'unverified',
            'geometry_not_worse_status':'unverified','acceptance_status':'unverified'}
    review=review or {}
    if review:
        if review.get('status') not in ['pass','fail','unverified']:raise ValueError('invalid review status')
        if review.get('status') in ['pass','fail'] and not all(review.get(k) for k in ['reviewer','evidence','rationale']):
            raise ValueError('manual review needs reviewer, evidence, rationale')
        result['manual_geometry_review']=review;result['geometry_not_worse_status']=review['status']
    if record.get('export_contract')!='formal_standard_glb':
        result.update(artifact_status='fail',acceptance_status='fail',error='unsupported/missing export frame contract; raw fast_glb/geometry_only not comparable')
        return result
    path=Path(record['glb'])
    if not path.is_file():
        result['acceptance_status']='fail';result['error']='required formal GLB missing';return result
    result['artifact_sha256']=sha256(path);result['artifact_bytes']=path.stat().st_size
    result['run_provenance']=verify_run_provenance(record,result['artifact_sha256'])
    result['provenance_status']=result['run_provenance']['status']
    if record.get('input'):
        ip=Path(record['input']);result['input_provenance']={'path':str(ip),'status':'present' if ip.is_file() else 'missing'}
        if ip.is_file():result['input_provenance']['sha256']=sha256(ip)
    try:
        scene=trimesh.load(path,force='scene',process=False)
        metrics=mesh_metrics(world_mesh(scene),reference_angle)
        result.update(artifact_status='pass',metrics=metrics,orientation_reference_status=metrics['orientation_reference_status'])
        # Ambiguous PCA never silently establishes orientation.
        # PCA 主次轴接近时不得默默给出朝向结论，保持 unverified。
        if metrics['surface_pca']['axis_ambiguous']:result['orientation_reference_status']='unverified'
        if result['orientation_reference_status']=='fail' or result['geometry_not_worse_status']=='fail':result['acceptance_status']='fail'
        elif result['orientation_reference_status']=='pass' and result['geometry_not_worse_status']=='pass' and result['provenance_status']=='verified':result['acceptance_status']='pass'
    except Exception as e:
        result.update(artifact_status='fail',acceptance_status='fail',error=f'{type(e).__name__}: {e}')
    if record.get('infer_log'):result['infer_log_provenance']=log_provenance(record['infer_log'])
    if record.get('command_file'):
        p=Path(record['command_file'])
        result['command_provenance']={'path':str(p),'status':'present' if p.is_file() else 'missing'}
        if p.is_file():
            command=p.read_text(encoding='utf-8',errors='replace');result['command_provenance'].update(sha256=sha256(p),command=command)
            if any(flag in command for flag in ['--fast_glb','--geometry_only','--ss_canonicalize','--ss_swap_yz']):
                result.update(artifact_status='fail',acceptance_status='fail',error='command contradicts fixed formal uncorrected frame contract')
    if result.get('provenance_status')=='fail':result.update(artifact_status='fail',acceptance_status='fail',error=result['run_provenance']['reason'])
    return result

def existing_manifest(root):
    """内置历史实验清单：base / old33k / axis10 三个 tag × 图片 001..004。
    只声明历史契约、GLB 与日志路径，不加载 checkpoint 张量；实际是否合规仍由上面的溯源校验判定。
    """
    root=Path(root).resolve();records=[]
    for tag in ['base','old33k','axis10']:
        for n in ['001','002','003','004']:
            if tag=='base':directory=root/'results_infer/ctrl_base_ss_datatest'/n
            elif tag=='old33k':directory=root/'results_infer/data_test_stage1_lora_33k_stage2_lora_20k'
            elif n=='001':directory=root/'results_infer/axis10_20k_001_recovery_20261013'
            else:directory=root/f'results_infer/axis10_20k_{n}_fullaudit_20261013'
            candidates=[directory/f'infer_{n}.log',directory/'infer.log']
            record={'id':f'{tag}_{n}','tag':tag,'image_id':n,'input':str(root/'data_test'/f'{n}.png'),
                    'glb':str(directory/f'{n}.glb'),'export_contract':'formal_standard_glb',
                    'declared_seed':42,'declared_pipeline_type':'1024','declared_axis_correction':False,
                    'declaration_source':'parent audited historical contract; independently inspect recorded command/log when available',
                    'checkpoint_contract':{'ss':str(root/'checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors') if tag=='base' else str(root/'results'/('multiview_1186_ss_flow_lora_64_views' if tag=='old33k' else 'multiview_1186_ss_flow_lora_64_canon_axis10')/'ckpts'/('denoiser_ema0.9999_step0033000.pt' if tag=='old33k' else 'denoiser_ema0.9999_step0020000.pt')),'shape':str(root/'results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt'),'decoder':str(root/'checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt'),'status':'historical audited declarations, actual load-path lines separately retained; checkpoint tensors not loaded here'},
                    'command_file':str(directory/'command.txt')}
            if tag=='old33k':candidates=[directory/f'infer_{n}.log',directory/f'{n}.infer.log',directory/'infer.log']
            record['infer_log']=str(next((p for p in candidates if p.exists()),candidates[0]))
            records.append(record)
    return {'schema':'trellis.fixed-frame-inputs.v1','records':records}

def run(manifest,output,reference_angle=10,reviews=None):
    """批量评估 manifest 中所有记录，写 <output>/report.json 与 input_manifest.json。
    聚合口径：任一 fail 则整体 fail；否则全部 pass 才 pass；其余为 unverified。
    """
    if not np.isfinite(reference_angle) or not 0<=reference_angle<=90:raise ValueError('angle must be finite 0..90')
    output=Path(output)
    # 输出目录必须不存在：避免覆盖上一次的验收报告。
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    reviews=reviews or {};results=[]
    ids=[r['id'] for r in manifest['records']]
    if len(set(ids))!=len(ids):raise ValueError('duplicate record ids')
    for record in manifest['records']:
        result=evaluate_record(record,reference_angle,reviews.get(record['id']));results.append(result)
        print(record['id'],result['artifact_status'],result['orientation_reference_status'],result['geometry_not_worse_status'],flush=True)
    report={'schema':'trellis.fixed-frame-report.v2','method':'CPU world-transformed fixed-frame AABB and exact uniform-triangle surface-integrated covariance PCA (primary); historical area-weighted face-center covariance retained as proxy; no mesh realignment/best-view search; topology component analysis not performed',
            'reference_orientation_threshold_degrees':reference_angle,
            'limitations':'10deg is a CLI diagnostic reference, not a newly authorized production acceptance rule. AABB/PCA/projected bbox are proxies, not geometry fidelity. Manual geometry review required against old baseline; absent review unverified. No automated image similarity score.',
            'results':results,'all_records_acceptance':'pass' if results and all(x['acceptance_status']=='pass' for x in results) else 'fail' if any(x['acceptance_status']=='fail' for x in results) else 'unverified'}
    (output/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report

def main():
    """CLI 入口：--root 与 --manifest 只能给一个；--manual-reviews 提供人工几何评审判定（pass/fail 需 reviewer+evidence+rationale）。"""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root');parser.add_argument('--manifest');parser.add_argument('--output',required=True);parser.add_argument('--reference-angle',type=float,default=10);parser.add_argument('--manual-reviews')
    args=parser.parse_args()
    if bool(args.root)==bool(args.manifest):parser.error('provide exactly one of --root or --manifest')
    manifest=json.loads(Path(args.manifest).read_text()) if args.manifest else existing_manifest(args.root)
    reviews=json.loads(Path(args.manual_reviews).read_text()) if args.manual_reviews else None
    run(manifest,args.output,args.reference_angle,reviews)

if __name__=='__main__':main()
