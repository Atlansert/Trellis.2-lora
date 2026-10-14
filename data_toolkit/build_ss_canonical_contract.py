#!/usr/bin/env python3
"""Stage a pose-constrained canonical dataset without touching old data.

Only existing native-alpha (symlink) objects are admitted by default. Derived
alpha stays quarantined until independently reviewed; native alpha is also
provisional, not a guarantee of semantic segmentation quality. No axis fixes.

中文说明：
- 用途：从已有数据集里筛出“姿态受限”的规范（canonical）子集，只读取、不修改旧数据。
- 输入：--root 数据根目录（需含 metadata.csv、renders_cond_t/、ss_latents_views_canon/、
  ss_view_voxels_canon_v4/），--output 新的输出目录（必须不存在），--max-angle 姿态角上限（默认 10）。
- 输出：output/canonical/<object>.npz（目标 latent）、manifest.json（逐对象记录+哈希）、
  summary.json（统计）、COMPLETE.json（标记 training_approved=False）。
- 关键约定：默认只接纳 native-alpha（符号链接）对象；derived alpha 一律隔离待人工复核；
  角度缓存只用于枚举候选，全部验收几何都由真实体素重算；不做任何坐标轴修正。
- 运行：python build_ss_canonical_contract.py --root <dir> --output <newdir> [--max-angle 10]
"""
import argparse, csv, hashlib, json, os
from pathlib import Path
import numpy as np
from PIL import Image
SCHEMA='trellis.canonical-contract.v1'

# 分块计算文件 SHA-256，用于 manifest 里的完整性/溯源校验。
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

# 由网格64体素坐标估计朝向：按既有编码器约定先置换成 (z,x,y)，再做 PCA，
# 返回夹角（度）、各轴跨度(+1 计数)、最大/次大特征值比。坐标非法时抛 ValueError。
def pose(coords):
    c=np.asarray(coords)
    if c.ndim!=2 or c.shape[1]!=3 or len(c)<4 or not np.isfinite(c).all():
        raise ValueError('invalid voxel coords')
    if not np.equal(c,np.floor(c)).all() or c.min()<0 or c.max()>63:
        raise ValueError('voxel coords outside integer grid64')
    c=c.astype(np.float64)[:,[2,0,1]] # Existing encoder convention; deliberately unchanged.
    ev,vec=np.linalg.eigh(np.cov(c,rowvar=False))
    if ev[-2]<=0: raise ValueError('degenerate PCA')
    return {'angle_degrees':float(np.degrees(np.arccos(np.clip(abs(vec[1,-1]),0,1)))),
            'spans':(np.ptp(c,axis=0)+1).astype(int).tolist(),
            'eigen_ratio':float(ev[-1]/ev[-2])}

# 姿态合格判据：偏航夹角 ≤ max_angle、最长跨度落在轴 1（长度轴）、特征值比 ≥ 3。
def valid_pose(p,max_angle):
    return p['angle_degrees']<=max_angle and int(np.argmax(p['spans']))==1 and p['eigen_ratio']>=3

# 读取 SS 均值 npz（要求 key='mean'），并强制校验形状 (8,16,16,16)、fp32、全有限。
def read_mean(path):
    with np.load(path,allow_pickle=False) as z:
        if 'mean' not in z: raise ValueError('required mean key missing')
        a=z['mean']
    if a.shape!=(8,16,16,16) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('invalid fp32 SS mean')
    return a

# 把帧记录里的相对路径安全解析为对象根目录下的真实文件；拒绝绝对路径与 .. 穿越。
def relative_frame(objroot,frame):
    rel=Path(frame['file_path'])
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe frame path')
    p=objroot/rel
    if not p.is_file():raise FileNotFoundError(p)
    return p

# 主流程：筛选合格对象 → 校验前 24 帧 yaw 序列、体素姿态与掩码 → 暂存目标 latent 与逐对象记录，
# 最后写 manifest.json / summary.json / COMPLETE.json。输出目录必须不存在（防止覆盖旧版本）。
def stage(root,output,max_angle):
    root=Path(root).resolve(); output=Path(output).resolve()
    if not 0<=max_angle<=90:raise ValueError('max angle must be finite 0..90')
    if output.exists():raise FileExistsError(output)
    # Fail on overwrite, even after interrupted staging: use a new version.
    output.mkdir(parents=True); cp=output/'canonical';cp.mkdir()
    with open(root/'metadata.csv',newline='') as f:
        rows=list(csv.DictReader(f))
    eligible={x['sha256'] for x in rows if x.get('cond_rendered','').lower()=='true' and float(x.get('aesthetic_score') or 0)>=4.5}
    render=root/'renders_cond_t'; latents=root/'ss_latents_views_canon';vox=root/'ss_view_voxels_canon_v4'
    cache=json.loads((render/'_view_offaxis.json').read_text())
    records=[]; exceptions=[]
    for obj in sorted(eligible):
        rr=render/obj; original_native=rr.is_symlink()
        entry={'object':obj,'mask_origin':'native_alpha_symlink' if original_native else 'derived_quarantined',
               'admitted':False,'views':[]}
        try:
            transform=rr/'transforms.json'; frames=json.loads(transform.read_text())['frames']
            entry['transforms_sha256']=digest(transform)
            if len(frames)<24 or len({f['file_path'] for f in frames[:24]})!=24:raise ValueError('invalid first24 frame sequence')
            for i,f in enumerate(frames[:24]):
                if f.get('is_real',False) or 'yaw_deg' not in f:raise ValueError('real/non-yaw frame in first24')
                if abs(float(f['yaw_deg'])-i*15)>1e-6:raise ValueError('unexpected yaw sequence')
                vp=vox/obj/f'{i:03d}.npz'
                # Cache is merely candidate enumeration. All acceptance geometry
                # is recomputed from the actual voxels; negative/rounded cache
                # values cannot approve a target.
                cache_angle=cache.get(obj,{}).get(Path(f['file_path']).name)
                if cache_angle is None or not np.isfinite(float(cache_angle)) or not 0<=float(cache_angle)<=90:
                    raise ValueError('missing/invalid angle cache entry')
                if float(cache_angle)>max_angle+0.01:continue
                with np.load(vp,allow_pickle=False) as z: geometry=pose(z['coords'])
                if not valid_pose(geometry,max_angle):continue
                ip=relative_frame(rr,f);lp=latents/obj/f'{i:03d}.npz';mean=read_mean(lp)
                with Image.open(ip) as im:a=np.asarray(im.convert('RGBA').getchannel('A'))
                strong=a>204
                mask_ok=bool(strong.any() and not np.all(a==255))
                ys,xs=np.nonzero(strong)
                v={'view':i,'file':f['file_path'],**geometry,'mask_nonempty_nonopaque':mask_ok,
                   'alpha_coverage':float(strong.mean()),'bbox':None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                   'image_sha256':digest(ip),'voxel_sha256':digest(vp),'latent_sha256':digest(lp)}
                entry['views'].append(v)
            candidates=[v for v in entry['views'] if v['mask_nonempty_nonopaque']]
            if not candidates:entry['reason']='no_pose_and_nonopaque_mask_candidate'
            else:
                best_angle=min(x['angle_degrees'] for x in candidates)
                # Opposite side views have essentially equal PCA axes; do not
                # let floating-point noise flip the target heading by 180deg.
                selected=min((x for x in candidates if x['angle_degrees']<=best_angle+0.05),key=lambda x:x['view'])
                entry['candidate_target_view']=selected['view']
                entry['input_views']=[x['view'] for x in candidates]
                if original_native:
                    source=latents/obj/f"{selected['view']:03d}.npz"
                    target=cp/f'{obj}.npz';np.savez_compressed(target,mean=read_mean(source))
                    if not np.array_equal(read_mean(source),read_mean(target)):raise ValueError('target changed')
                    entry['target_sha256']=digest(target);entry['target_source_sha256']=digest(source)
                    entry['admitted']=True;entry['reason']='provisional_native_alpha_requires_axis_validation'
                else:entry['reason']='derived_alpha_requires_independent_review'
        except Exception as e:
            entry['error']=f'{type(e).__name__}: {e}';exceptions.append(entry['object'])
        records.append(entry)
        if len(records)%100==0:print('checked',len(records),flush=True)
    admitted=[x for x in records if x['admitted']]
    manifest={'schema':SCHEMA,'status':'staged_not_training_approved','root':str(root),'max_angle_degrees':max_angle,
              'encoder_permutation':[2,0,1],'mask_policy':'native_alpha_only; derived quarantined; semantic quality not guaranteed',
              'selection':'independent PCA<=threshold; within 0.05deg of minimum choose lowest original frame index to stabilize opposite-view ties; never aspect argmax or fallback',
              'metadata_sha256':digest(root/'metadata.csv'),'cache_sha256':digest(render/'_view_offaxis.json'),
              'builder_sha256':digest(Path(__file__)),'objects':records}
    summary={'eligible_objects':len(records),'admitted_provisional_objects':len(admitted),
             'input_pairs':sum(len(x['input_views']) for x in admitted),'quarantined_objects':len(records)-len(admitted),
             'error_objects':exceptions,'max_target_angle':max((next(v['angle_degrees'] for v in x['views'] if v['view']==x['candidate_target_view']) for x in admitted),default=None),
             'extreme_object_admitted':any(x['object']=='3e988013c8136c9b' for x in admitted),
             'invariants':'all targets from accepted input set, finite fp32 mean shape8x16x16x16; angle<=10 and longindex1; old data untouched'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if exceptions:raise RuntimeError(f'{len(exceptions)} object errors; staging not approved')
    (output/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':digest(output/'manifest.json'),'summary_sha256':digest(output/'summary.json'),'training_approved':False},indent=2),encoding='utf-8')

# CLI：--root 数据根目录，--output 新输出目录，--max-angle 允许的最大姿态角（默认 10 度）。
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--max-angle',type=float,default=10)
    args=p.parse_args()
    stage(args.root,args.output,args.max_angle)
