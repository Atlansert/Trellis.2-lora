#!/usr/bin/env python3
"""
PBR 数据管线驱动 (自包含, 替代官方依赖 datasets/ 模块的脚本链):
  Stage A: Blender PBR dump   (并行)   -> pbr_dumps/{sha}.pickle
  Stage B: o_voxel 体素化      (并行)   -> pbr_voxels_1024/{sha}.vxz
  Stage C: tex_enc 编码       (GPU)    -> pbr_latents/tex_enc_next_dc_f16c32_fp16_1024/{sha}.npz
  Stage D: 合并 metadata                  -> training_data/multiview_1186/metadata.csv

用法:
  python scripts/prepare_pbr_pipeline.py --stage A          # dump
  python scripts/prepare_pbr_pipeline.py --stage B          # voxelize
  python scripts/prepare_pbr_pipeline.py --stage C          # encode (需要 GPU)
  python scripts/prepare_pbr_pipeline.py --stage D          # merge metadata
"""
import os, sys, json, argparse, subprocess, glob, csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import torch
import pandas as pd

ROOT = '/data/lzx/TRELLIS.2'
DATA = os.path.join(ROOT, 'training_data/multiview_1186')
BLENDER = '/tmp/blender-4.5.1-linux-x64/blender'
DUMP_SCRIPT = os.path.join(ROOT, 'data_toolkit/blender_script/dump_pbr.py')
PBR_DUMPS = os.path.join(DATA, 'pbr_dumps')
VOXELS = os.path.join(DATA, 'pbr_voxels_1024')
LATENTS = os.path.join(DATA, 'pbr_latents/tex_enc_next_dc_f16c32_fp16_1024')
RES = 1024
META = os.path.join(DATA, 'metadata.csv')


def load_meta():
    df = pd.read_csv(META)
    return df


def finished(sha, kind):
    if kind == 'dump':
        return os.path.exists(os.path.join(PBR_DUMPS, f'{sha}.pickle'))
    if kind == 'voxel':
        return os.path.exists(os.path.join(VOXELS, f'{sha}.vxz'))
    if kind == 'latent':
        return os.path.exists(os.path.join(LATENTS, f'{sha}.npz'))
    return False


def stage_a(workers=8):
    """Blender PBR dump, skip already-done. One Blender process per asset."""
    df = load_meta()
    todo = [r for _, r in df.iterrows() if not finished(r['sha256'], 'dump')]
    print(f'[A] dump: {len(todo)} 待处理 / {len(df)} 总', flush=True)
    paths = [(r['sha256'], os.path.join(ROOT, r['local_path'])) for r in todo]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.map(lambda p: dump_one(*p), paths)
    done = sum(1 for _, r in df.iterrows() if finished(r['sha256'], 'dump'))
    print(f'[A] done: {done}/{len(df)}', flush=True)


def dump_one(sha, obj_path):
    out = os.path.join(PBR_DUMPS, f'{sha}.pickle')
    tmp = out + '.tmp'
    try:
        subprocess.run([BLENDER, '-b', '-P', DUMP_SCRIPT, '--',
                        '--object', obj_path, '--output_path', tmp],
                       capture_output=True, timeout=600)
        if os.path.exists(tmp):
            os.replace(tmp, out)
        else:
            with open(out + '.err', 'w') as f:
                f.write('blender produced no output')
    except Exception as e:
        with open(out + '.err', 'w') as f:
            f.write(str(e))


def vox_one(path):
    import o_voxel
    sha = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(VOXELS, f'{sha}.vxz')
    if os.path.exists(out_path):
        return
    import pickle as pkl
    with open(path, 'rb') as f:
        dump = pkl.load(f)
    dump = fix_dump(dump)
    if dump is None:
        return
    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump, grid_size=RES, aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], mip_level_offset=0)
    keep = {'base_color': attr['base_color'], 'metallic': attr['metallic'],
            'roughness': attr['roughness'], 'alpha': attr['alpha']}
    # 对齐至 dual_grid coords (shape latent 同源), 保证 tex/shape 两流 coords 完全一致,
    # 否则 SLatPbr 数据集断言 torch.equal(shape_z.coords, pbr_z.coords) 会失败。
    sys.path.insert(0, os.path.dirname(__file__))
    target = align_to_dual_grid(sha, coord, keep)
    if target is None:
        # 无 dual_grid vxz 的资产: 退回原始体素坐标 (训练时会被数据集过滤)
        target = (coord, keep)
    o_voxel.io.write_vxz(out_path, target[0], target[1])


def align_to_dual_grid(sha, coord, attr):
    """把 pbr 光栅化体素属性按最近邻映射到 dual_grid_1024 的体素坐标上。"""
    import o_voxel
    from scipy.spatial import cKDTree
    dg_path = os.path.join(DATA, 'dual_grid_1024', f'{sha}.vxz')
    if not os.path.exists(dg_path):
        return None
    dg_coords, _ = o_voxel.io.read_vxz(dg_path, num_threads=4)
    dg_coords = dg_coords.numpy().astype(np.int64)
    src_coords = coord.numpy().astype(np.int64)
    tree = cKDTree(src_coords.astype(np.float64))
    _, idx = tree.query(dg_coords.astype(np.float64), k=1, workers=4)
    return (
        torch.tensor(dg_coords, dtype=torch.int32),
        {k: v[idx] for k, v in attr.items()},
    )


def fix_dump(dump):
    import numpy as np
    from copy import deepcopy
    dump = deepcopy(dump)
    for m in dump.get('materials', []):
        if m.get('alphaTexture') is not None and m.get('alphaMode') == 'OPAQUE':
            m['alphaMode'] = 'BLEND'
    # 官方 voxelize_pbr 的做法: 追加一个默认灰色材质给无材质面 (mat_ids == -1)，
    # 否则 o_voxel 对这些面做材质采样时除零 -> SIGFPE
    materials = dump.get('materials', [])
    if materials:
        default_idx = len(materials)
        materials.append({
            'baseColorFactor': [0.5, 0.5, 0.5, 1.0],
            'metallicFactor': 0.0,
            'roughnessFactor': 0.5,
            'alphaFactor': 1.0,
            'alphaMode': 'OPAQUE',
            'alphaCutoff': 0.5,
            'baseColorTexture': None,
            'metallicTexture': None,
            'roughnessTexture': None,
            'alphaTexture': None,
        })
    else:
        default_idx = 0
    fixed_objects = []
    for o in dump.get('objects', []):
        if not isinstance(o, dict) or 'vertices' not in o:
            continue
        v = np.asarray(o['vertices'], dtype=np.float64)
        f = np.asarray(o.get('faces', []))
        if len(v) == 0 or len(f) == 0:
            continue
        # 剔除退化三角形 (零面积面片会让 o_voxel 光栅化除零 -> SIGFPE)
        t = v[f]
        cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area = np.linalg.norm(cross, axis=1)
        keep = area > 1e-12
        if not keep.any():
            continue
        o['faces'] = f[keep]
        o['normals'] = np.asarray(o['normals'])[keep]
        o['uvs'] = np.asarray(o['uvs'])[keep]
        mat_ids = np.asarray(o['mat_ids'])[keep].astype(np.int64)
        mat_ids[mat_ids < 0] = default_idx
        o['mat_ids'] = mat_ids
        center = (v.min(0) + v.max(0)) / 2
        span = (v.max(0) - v.min(0)).max()
        scale = 1.0 / span if span > 0 else 1.0
        v = (v - center) * scale
        # 轴修正: 与 mesh_dumps/dual_grid 对齐 (pbr dump 的 y/z 轴与 dual grid 源互换)
        v = v[:, [0, 2, 1]]
        o['vertices'] = v.astype(np.float32)
        fixed_objects.append(o)
    if not fixed_objects:
        return None
    dump['objects'] = fixed_objects
    return dump


def stage_b(workers=16):
    df = load_meta()
    todo = [r['sha256'] for _, r in df.iterrows() if finished(r['sha256'], 'dump') and not finished(r['sha256'], 'voxel')]
    print(f'[B] voxelize: {len(todo)} 待处理', flush=True)
    paths = [os.path.join(PBR_DUMPS, f'{s}.pickle') for s in todo]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        ex.map(vox_one, paths)
    done = sum(1 for r in df['sha256'] if finished(r, 'voxel'))
    print(f'[B] done: {done}/{len(df)}', flush=True)


def stage_c(gpu='5'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import torch
    sys.path.insert(0, ROOT)
    import o_voxel
    import trellis2.models as models
    import trellis2.modules.sparse as sp

    df = load_meta()
    todo = [r['sha256'] for _, r in df.iterrows() if finished(r['sha256'], 'voxel') and not finished(r['sha256'], 'latent')]
    print(f'[C] encode: {len(todo)} 待处理', flush=True)
    encoder = models.from_pretrained(os.path.join(ROOT, 'checkpoints/ckpts/tex_enc_next_dc_f16c32_fp16')).cuda().eval()

    with torch.no_grad():
        for i, sha in enumerate(todo):
            coords, attr = o_voxel.io.read_vxz(os.path.join(VOXELS, f'{sha}.vxz'), num_threads=4)
            feats_parts = [attr[k] for k in ('base_color', 'metallic', 'roughness', 'alpha')]
            feats = torch.cat(feats_parts, dim=-1).float() / 255.0 * 2 - 1
            x = sp.SparseTensor(
                feats,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1).int(),
            )
            z = encoder(x.cuda())
            out = {
                'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8),
                'feats': z.feats.cpu().numpy().astype(np.float32),
            }
            os.makedirs(LATENTS, exist_ok=True)
            np.savez_compressed(os.path.join(LATENTS, f'{sha}.npz'), **out)
            if (i + 1) % 50 == 0:
                print(f'[C] {i+1}/{len(todo)}', flush=True)
    done = sum(1 for r in df['sha256'] if finished(r, 'latent'))
    print(f'[C] done: {done}/{len(df)}', flush=True)


def stage_d():
    df = load_meta()
    df['pbr_dumped'] = df['sha256'].apply(lambda s: finished(s, 'dump'))
    df['pbr_voxelized'] = df['sha256'].apply(lambda s: finished(s, 'voxel'))
    df['pbr_latent_encoded'] = df['sha256'].apply(lambda s: finished(s, 'latent'))
    tokens = []
    for sha in df['sha256']:
        p = os.path.join(LATENTS, f'{sha}.npz')
        if os.path.exists(p):
            tokens.append(len(np.load(p)['coords']))
        else:
            tokens.append(None)
    df['pbr_latent_tokens'] = tokens
    df.to_csv(META, index=False)
    enc = int(df['pbr_latent_encoded'].sum())
    valid = [t for t in tokens if t is not None]
    print(f'[D] metadata 更新: pbr_latent_encoded={enc}/{len(df)}')
    if valid:
        print(f'    平均 tokens: {np.mean(valid):.0f} (基于 {len(valid)} 个已编码)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['A', 'B', 'C', 'D'])
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--gpu', type=str, default='5')
    args = ap.parse_args()
    for d in (PBR_DUMPS, VOXELS, LATENTS):
        os.makedirs(d, exist_ok=True)
    if args.stage == 'A':
        stage_a(args.workers or 8)
    elif args.stage == 'B':
        stage_b(args.workers or 16)
    elif args.stage == 'C':
        stage_c(args.gpu)
    elif args.stage == 'D':
        stage_d()