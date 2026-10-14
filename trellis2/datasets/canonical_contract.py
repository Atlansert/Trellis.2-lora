"""Immutable canonical candidate contract validation (stdlib only).

Optional for legacy data; enabled contracts fail closed. Staged data may be
opened only through explicit diagnostics opt-in, never by default training.
"""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_canonical_contract(manifest_path, canonical_root, allow_staged=False):
    if canonical_root is None:
        raise ValueError('canonical root required for contract')
    manifest_path=Path(manifest_path).resolve()
    parent=manifest_path.parent
    complete=json.loads((parent/'COMPLETE.json').read_text())
    manifest_hash=file_sha256(manifest_path)
    if complete.get('manifest_sha256')!=manifest_hash:
        raise ValueError('canonical manifest checksum mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='trellis.canonical-contract.v1':
        raise ValueError('unsupported canonical contract schema')
    if Path(canonical_root).resolve()!=(parent/'canonical').resolve():
        raise ValueError('canonical root is not contract output')
    if not allow_staged:
        approval_path=parent/'TRAINING_APPROVAL.json'
        if not approval_path.is_file():
            raise ValueError('staged canonical data not approved for training')
        approval=json.loads(approval_path.read_text())
        if approval.get('manifest_sha256')!=manifest_hash or approval.get('approved') is not True:
            raise ValueError('canonical training approval invalid')
        # Approval must reference immutable successful geometry + mask reviews,
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
    root=Path(manifest['root']).resolve()
    for path,key in [(root/'metadata.csv','metadata_sha256'),(root/'renders_cond_t/_view_offaxis.json','cache_sha256')]:
        if file_sha256(path)!=manifest.get(key):
            raise ValueError(f'canonical contract dependency changed: {path}')
    admitted={}
    for obj in manifest['objects']:
        if not obj['admitted']:continue
        name=obj['object']
        if name in admitted or Path(name).name!=name or name in ('.','..'):
            raise ValueError('invalid/duplicate object id')
        frames=json.loads((root/'renders_cond_t'/name/'transforms.json').read_text())['frames']
        if file_sha256(root/'renders_cond_t'/name/'transforms.json')!=obj['transforms_sha256']:
            raise ValueError(f'canonical frame sequence changed: {name}')
        views=obj['input_views']
        if not views or obj['candidate_target_view'] not in views:
            raise ValueError('canonical target not in accepted input set')
        if file_sha256(parent/'canonical'/f'{name}.npz')!=obj['target_sha256']:
            raise ValueError(f'canonical target changed: {name}')
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
    if not admitted:raise ValueError('canonical contract admits no objects')
    return root,admitted
