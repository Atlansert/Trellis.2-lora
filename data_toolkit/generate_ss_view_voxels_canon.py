#!/usr/bin/env python3
"""逐视角 SS 体素化 —— **修正相机约定**版（2026-10-11）。

背景（实测，见 REPORT §11）：
`generate_ss_view_voxels_mesh_t1.py` 直接对原始 mesh dump 顶点做
`v @ rz(yaw-270)`，即绕 dump 帧的 **Z** 轴旋转。但渲染侧
（`render_view.py` / `render_cond.py`）的相机是在 **XY 平面环绕**、
以 **+Z 为竖直**、并把相机 local Y 对齐世界竖直轴——两者对"哪根轴是环绕轴"
的理解不一致。实测后果：

  占 71% 的对象（网格长度轴 = dump 的 Z）→ 体素化等于**绕自身长度轴自转**，
  12 个视角的 latent 几乎完全相同，而渲染图是正常环绕（侧视 6.9 → 端视 0.9）。
  另外 29% 的对象（长度轴 = X）碰巧在正确的平面内扫动。

修正：**先把 dump 帧映射到渲染帧（交换 y/z），再绕 Z 旋转**，于是旋转成为真正的环绕。
验证（与 `renders_cond_t` 的剪影宽高比逐视角对照，两类对象均吻合）：

    view      0     2     4     6     8    10
    渲染    6.91  5.30  3.06  0.90  3.06  5.17
    swap(E0/E2) 6.4  5.9   3.7   0.8   3.7   5.9

得到的网格轴语义：index0 = 画面水平、index1 = 视深、index2 = 画面竖直
（index2 的跨度在所有视角恒为对象高度，符合"竖直不受环绕影响"）。

⚠️ 仅交换 y/z 会引入**镜像**（对换矩阵行列式 -1）。用
`scripts/utils/canon_transform_search.py` 在 16 个候选（轴映射 × 符号 × 环绕偏移）
上以"GT 剪影 vs 渲染剪影 IoU"搜索，最优组合为

    轴映射 (0,2,1) 即 (x,y,z)_D -> (x,z,y)   符号 (-1,+1,-1)   偏移 90

未翻转 IoU = 0.848（翻转版仅 0.776，手性正确）；而只做 y/z 交换时为 0.770 且
翻转版更高（0.785），即确实是镜像。该组合是纯旋转（符号乘积 = +1）。

用法:
    python generate_ss_view_voxels_canon.py --root training_data/multiview_1186 \
        --output_name ss_view_voxels_canon_v3 --num_views 24
"""
import argparse
import glob
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import open3d as o3d


def rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def one(job):
    root, obj, out_name, res, num_views, overwrite, axis_map, signs, yaw_offset, perm = job
    dump_path = os.path.join(root, 'mesh_dumps', obj + '.pickle')
    render_dir = os.path.join(root, 'renders_cond', obj)
    out_dir = os.path.join(root, out_name, obj)
    os.makedirs(out_dir, exist_ok=True)
    try:
        frames = json.load(open(os.path.join(render_dir, 'transforms.json')))['frames'][:num_views]
        dump = pickle.load(open(dump_path, 'rb'))
    except Exception as e:
        return obj, 0, str(e)
    vertices, faces, offset = [], [], 0
    for part in dump['objects']:
        v = np.asarray(part['vertices'])
        f = np.asarray(part['faces'])
        if len(v):
            vertices.append(v)
            faces.append(f.astype(np.int64) + offset)
            offset += len(v)
    if not vertices:
        return obj, 0, 'empty mesh'
    vertices = np.concatenate(vertices).astype(np.float64)
    faces = np.concatenate(faces) if faces else None
    # ★ 修正点：先映射到渲染帧（y/z 交换 + 符号），使 rz 成为真正的环绕，
    # 且保持手性（纯旋转）。符号与偏移由 canon_transform_search.py 标定。
    if axis_map is not None:
        vertices = vertices[:, axis_map]
    vertices = vertices * np.array(signs, dtype=np.float64)
    done = 0
    for i, fr in enumerate(frames):
        path = os.path.join(out_dir, f'{i:03d}.npz')
        if os.path.exists(path) and not overwrite:
            done += 1
            continue
        yaw = float(fr.get('yaw_deg', i * 15.0))
        vv = vertices @ rz(yaw - yaw_offset)   # 注意：不要用 offset，那是顶点索引累加器
        if faces is not None and len(faces):
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vv),
                o3d.utility.Vector3iVector(faces))
            grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh, voxel_size=1 / res,
                min_bound=(-.5, -.5, -.5), max_bound=(.5, .5, .5))
            coords = np.array([v.grid_index for v in grid.get_voxels()], dtype=np.uint8)
        else:
            coords = np.unique(np.clip(np.floor((vv + .5) * res), 0, res - 1), axis=0).astype(np.uint8)
        if perm is not None and len(coords):
            coords = coords[:, perm]
        if len(coords):
            coords = np.clip(coords, 0, res - 1)
            # 去重（置换后仍可能重复，虽然不会）
            coords = np.unique(coords, axis=0)
        np.savez_compressed(path, coords=coords)
        done += 1
    return obj, done, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output_name', default='ss_view_voxels_canon')
    ap.add_argument('--num_views', type=int, default=24)
    ap.add_argument('--resolution', type=int, default=64)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--axis_map', default='0,2,1',
                    help='dump 顶点的轴映射，默认标定值 "0,2,1"')
    ap.add_argument('--signs', default='-1,1,-1',
                    help='dump 顶点取反，默认标定值 "-1,1,-1"（保证是纯旋转）')
    ap.add_argument('--offset', type=float, default=90.0,
                    help='环绕角偏移，默认标定值 90')
    ap.add_argument('--perm', default='',
                    help='可选：对**体素输出**再做一次轴置换（一般不用，须留空）')
    a = ap.parse_args()
    # 注意：axis_map 作用于顶点（与旋转合成），perm 作用于体素输出，二者独立。
    # 早期版本把同一个 --perm 同时用于两处，导致置换两次，务必区分。
    axis_map = [int(x) for x in a.axis_map.split(',')] if a.axis_map else None
    perm = [int(x) for x in a.perm.split(',')] if a.perm else None
    signs = [float(x) for x in a.signs.split(',')]
    offset = a.offset

    root = os.path.abspath(a.root)
    os.makedirs(os.path.join(root, a.output_name), exist_ok=True)
    objs = [os.path.basename(p)[:-7] for p in glob.glob(os.path.join(root, 'mesh_dumps', '*.pickle'))
            if os.path.isdir(os.path.join(root, 'renders_cond', os.path.basename(p)[:-7]))]
    print(f'objects={len(objs)} views={len(objs) * a.num_views} '
          f'output={os.path.join(root, a.output_name)} axis_map={axis_map} '
          f'signs={signs} offset={offset} perm={perm}', flush=True)
    jobs = [(root, o, a.output_name, a.resolution, a.num_views, a.overwrite,
             axis_map, signs, offset, perm) for o in sorted(objs)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (o, d, e) in enumerate(ex.map(one, jobs), 1):
            if e:
                print(f'{o}: {e}', flush=True)
            if n % 50 == 0 or n == len(jobs):
                print(f'{n}/{len(jobs)} ({d}/{a.num_views})', flush=True)


if __name__ == '__main__':
    main()