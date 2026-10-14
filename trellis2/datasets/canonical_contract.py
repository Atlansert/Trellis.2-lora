"""Immutable canonical candidate contract validation (stdlib only).

Optional for legacy data; enabled contracts fail closed. Staged data may be
opened only through explicit diagnostics opt-in, never by default training.

中文：canonical 数据的"不可变契约"校验（只用标准库）。
职责：用 SHA256 锁定候选 canonical 目录的清单（manifest）、依赖数据（metadata.csv、
视角缓存）、每个物体的帧序列与各视角源文件，并校验训练审批证据。
输入：manifest_path、canonical_root、allow_staged；输出：(root, admitted) 已通过校验的物体字典。
契约：默认 fail-closed（任何不一致都抛 ValueError）；staged（未审批）数据必须显式
allow_staged 才能打开；审批必须引用可校验且 passed 的几何/mask 证据文件，而不是一个布尔开关。
"""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    """分块读取文件并返回其 SHA256 十六进制摘要（避免一次性读入大文件）。"""
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_canonical_contract(manifest_path, canonical_root, allow_staged=False):
    """校验 canonical 契约，返回 (canonical_root, admitted 物体字典)；任何不一致都抛 ValueError。"""
    if canonical_root is None:
        raise ValueError('canonical root required for contract')
    manifest_path=Path(manifest_path).resolve()
    parent=manifest_path.parent
    # COMPLETE.json 中记录的清单哈希必须与当前清单文件一致（清单不可被事后修改）
    complete=json.loads((parent/'COMPLETE.json').read_text())
    manifest_hash=file_sha256(manifest_path)
    if complete.get('manifest_sha256')!=manifest_hash:
        raise ValueError('canonical manifest checksum mismatch')
    manifest=json.loads(manifest_path.read_text())
    # 只接受已知的 schema 版本
    if manifest.get('schema')!='trellis.canonical-contract.v1':
        raise ValueError('unsupported canonical contract schema')
    # canonical_root 必须正好是该契约目录下的 canonical 输出，防止指向别处
    if Path(canonical_root).resolve()!=(parent/'canonical').resolve():
        raise ValueError('canonical root is not contract output')
    # 未显式允许 staged 时：必须有与清单哈希绑定的训练审批，且几何/mask 证据校验通过
    if not allow_staged:
        approval_path=parent/'TRAINING_APPROVAL.json'
        if not approval_path.is_file():
            raise ValueError('staged canonical data not approved for training')
        approval=json.loads(approval_path.read_text())
        if approval.get('manifest_sha256')!=manifest_hash or approval.get('approved') is not True:
            raise ValueError('canonical training approval invalid')
        # Approval must reference immutable successful geometry + mask reviews,
        # 中文：审批必须引用不可变且已通过的几何/mask 评审证据，而不是清单里的一个布尔位
        # not merely a boolean toggled in the candidate manifest.
        for key in ('coordinate_evidence','mask_evidence'):
            evidence=approval.get(key,{})
            relative=Path(evidence.get('path',''))
            if relative.is_absolute() or '..' in relative.parts or not evidence.get('sha256'):
                raise ValueError(f'missing/unsafe approval {key}')
            ep=parent/relative
            if not ep.is_file() or file_sha256(ep)!=evidence['sha256']:
                raise ValueError(f'approval evidence checksum mismatch: {key}')
            report=json.loads(ep.read_text())
            if report.get('passed') is not True:
                raise ValueError(f'approval evidence did not pass: {key}')
    # 依赖文件哈希校验：metadata.csv 与视角缓存一变，整个契约即失效
    root=Path(manifest['root']).resolve()
    for path,key in [(root/'metadata.csv','metadata_sha256'),(root/'renders_cond_t/_view_offaxis.json','cache_sha256')]:
        if file_sha256(path)!=manifest.get(key):
            raise ValueError(f'canonical contract dependency changed: {path}')
    # 逐个物体校验，只接受清单中标记 admitted 的物体
    admitted={}
    for obj in manifest['objects']:
        if not obj['admitted']:continue
        name=obj['object']
        # 物体名必须是安全 basename 且不重复
        if name in admitted or Path(name).name!=name or name in ('.','..'):
            raise ValueError('invalid/duplicate object id')
        frames=json.loads((root/'renders_cond_t'/name/'transforms.json').read_text())['frames']
        if file_sha256(root/'renders_cond_t'/name/'transforms.json')!=obj['transforms_sha256']:
            raise ValueError(f'canonical frame sequence changed: {name}')
        # 目标视图必须属于被接受的输入视图集合
        views=obj['input_views']
        if not views or obj['candidate_target_view'] not in views:
            raise ValueError('canonical target not in accepted input set')
        if file_sha256(parent/'canonical'/f'{name}.npz')!=obj['target_sha256']:
            raise ValueError(f'canonical target changed: {name}')
        # 逐视图校验：角度/mask 有效、帧序号与 file_path 一致、图像/体素/潜变量三类源文件哈希一致
        valid_views=[]
        for view in obj['views']:
            if view['view'] not in views:continue
            if not 0<=view['angle_degrees']<=manifest['max_angle_degrees'] or not view['mask_nonempty_nonopaque']:
                raise ValueError('invalid accepted view geometry/mask')
            index=view['view'];relative=Path(view['file'])
            if relative.is_absolute() or '..' in relative.parts or not 0<=index<24:
                raise ValueError('unsafe accepted frame')
            if frames[index]['file_path']!=view['file']:
                raise ValueError('frame index changed')
            paths=[(root/'renders_cond_t'/name/relative,'image_sha256'),
                   (root/'ss_view_voxels_canon_v4'/name/f'{index:03d}.npz','voxel_sha256'),
                   (root/'ss_latents_views_canon'/name/f'{index:03d}.npz','latent_sha256')]
            for path,key in paths:
                if file_sha256(path)!=view[key]:raise ValueError(f'canonical source changed: {path}')
            valid_views.append(index)
        if valid_views!=views:raise ValueError('accepted views missing or out of order')
        admitted[name]=obj
    # 一个物体都没通过 => 数据不可用（fail closed）
    if not admitted:raise ValueError('canonical contract admits no objects')
    return root,admitted
