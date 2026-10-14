#!/usr/bin/env python3
"""
不依赖 Blender 的自定义数据预处理脚本
=============================================
将 3D 模型 (+ 可选图片) 的训练数据转换为 TRELLIS.2 shape 训练所需的格式:
  1. trimesh 加载模型 (GLB/OBJ/PLY/STL 等) → 归一化 mesh
  2. o_voxel API 转 dual grid .vxz (256, 512, 1024)
  3. 可选: 复制图片到 renders_cond/{sha256}/ + 生成 transforms.json
  4. 生成 metadata.csv

支持两种数据目录结构:

  【子目录模式 (默认)】每个子目录包含模型 [+ 图片]:
      data_dir/
        ├── sample_001/
        │   ├── model.glb       (或 .obj)
        │   └── image.png       (可选)
        └── sample_002/
            └── model.obj

  【Flat 模式 (--flat)】路径下直接是模型文件，图片按文件名 stem 匹配:
      data_dir/
        ├── train_001.glb
        ├── train_002.obj
        ├── train_001.png       (可选, 按 stem 匹配 train_001.glb)
        └── train_002.jpg       (可选, 按 stem 匹配 train_002.obj)

用法:
    cd /home/lyx/lzx/TRELLIS.2

    # 子目录模式
    python data_toolkit/custom_prepare_data.py \
        --data_dir ./week10_data \
        --output_dir ./training_data/custom_dataset

    # Flat 模式
    python data_toolkit/custom_prepare_data.py \
        --data_dir ./my_models \
        --output_dir ./training_data/custom_dataset \
        --flat
"""

import os
import sys
import json
import pickle
import hashlib
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict

# trimesh 在两种模式都需要；o_voxel 只在完整预处理时需要 GPU。
# 保持条件图刷新模式不初始化 CUDA，便于在登录/调度节点维护真实视图。
import trimesh


def sha256_file(path: str) -> str:
    """计算文件的 SHA-256"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]  # 用前16位，够唯一了


def load_and_normalize_mesh(mesh_path: str) -> tuple:
    """
    用 trimesh 加载 3D 模型 (GLB/OBJ/PLY/STL 等)，合并所有 sub-mesh，归一化到 [-0.5, 0.5]³
    返回: (vertices: torch.Tensor [V,3], faces: torch.Tensor [F,3])
    """
    scene_or_mesh = trimesh.load(mesh_path, force='mesh')

    if isinstance(scene_or_mesh, trimesh.Scene):
        vertices_list, faces_list = [], []
        offset = 0
        for geom in scene_or_mesh.geometry.values():
            if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                v = np.array(geom.vertices, dtype=np.float32)
                f = np.array(geom.faces, dtype=np.int64)
                if len(v) > 0 and len(f) > 0:
                    vertices_list.append(v)
                    faces_list.append(f + offset)
                    offset += len(v)
        if not vertices_list:
            raise ValueError("模型文件中没有有效的 mesh 几何体")
        vertices = np.concatenate(vertices_list, axis=0)
        faces = np.concatenate(faces_list, axis=0)
    else:
        vertices = np.array(scene_or_mesh.vertices, dtype=np.float32)
        faces = np.array(scene_or_mesh.faces, dtype=np.int64)

    vertices = torch.from_numpy(vertices).float()
    faces = torch.from_numpy(faces).long()

    # 归一化: center + scale 到 [-0.5, 0.5]³
    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    center = (v_min + v_max) / 2
    scale = 0.99999 / (v_max - v_min).max().clamp(min=1e-8)
    vertices = (vertices - center) * scale

    assert torch.all(vertices >= -0.5001) and torch.all(vertices <= 0.5001), \
        f"归一化失败: range=[{vertices.min():.4f}, {vertices.max():.4f}]"

    return vertices, faces


def convert_to_dual_grid(vertices, faces, sha256, output_dir, resolutions=[256, 512, 1024]):
    """
    将归一化 mesh 转为 o_voxel dual grid 格式
    """
    results = {}
    for res in resolutions:
        dual_dir = os.path.join(output_dir, f'dual_grid_{res}')
        os.makedirs(dual_dir, exist_ok=True)
        vxz_path = os.path.join(dual_dir, f'{sha256}.vxz')

        if os.path.exists(vxz_path):
            try:
                info = o_voxel.io.read_vxz_info(vxz_path)
                results[res] = info['num_voxel']
                print(f"  [skip] dual_grid_{res} 已存在 ({info['num_voxel']} voxels)")
                continue
            except Exception:
                pass  # 文件损坏，重新生成

        print(f"  生成 dual_grid_{res}...")
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=vertices,
            faces=faces,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=False,
        )

        # 量化 dual_vertices 到 [0, 255] uint8
        dual_vertices = dual_vertices * res - voxel_indices
        dual_vertices = torch.clamp(dual_vertices, 0, 1)
        dual_vertices = (dual_vertices * 255).to(torch.uint8)

        # 合并 intersected flags
        intersected = (
            intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]
        ).to(torch.uint8)

        o_voxel.io.write_vxz(
            vxz_path,
            voxel_indices,
            {'vertices': dual_vertices, 'intersected': intersected},
        )

        results[res] = len(dual_vertices)
        print(f"  dual_grid_{res}: {len(dual_vertices)} voxels → {vxz_path}")

    return results


def create_mesh_dump(vertices, faces, sha256, output_dir):
    """
    创建 mesh dump pickle 文件，用于 Shape VAE 训练时的渲染 loss 计算

    格式与 dump_mesh.py (Blender脚本) 兼容:
      {
          'objects': [
              {'vertices': np.ndarray [V,3], 'faces': np.ndarray [F,3]},
              ...
          ]
      }
    """
    mesh_dir = os.path.join(output_dir, 'mesh_dumps')
    os.makedirs(mesh_dir, exist_ok=True)
    pickle_path = os.path.join(mesh_dir, f'{sha256}.pickle')

    if os.path.exists(pickle_path):
        print(f"  [skip] mesh_dump 已存在")
        return pickle_path

    dump = {
        'objects': [{
            'vertices': vertices.numpy().astype(np.float32),
            'faces': faces.numpy().astype(np.int32),
        }]
    }

    with open(pickle_path, 'wb') as f:
        pickle.dump(dump, f)

    size_kb = os.path.getsize(pickle_path) / 1024
    print(f"  mesh_dump: {len(vertices)} verts, {len(faces)} faces → {pickle_path} ({size_kb:.1f} KB)")

    return pickle_path


def _parse_multiview_pngs(image_paths):
    """
    解析多视角图片文件列表，提取渲染角度信息。

    支持两种命名模式:
      Pattern A: {idx:03d}_cam_{angle:03d}.png  → 角度从文件名提取
      Pattern B: {idx:03d}.png                   → 按 15° 间隔推算 (idx*15)

    返回: [(dst_filename, angle_deg, src_path), ...] 或 None（无法解析时）
    """
    if len(image_paths) <= 1:
        return None

    # Pattern A: *_cam_*.png
    cam_images = {}
    for p in image_paths:
        stem = Path(p).stem
        if '_cam_' in stem:
            parts = stem.split('_cam_')
            if len(parts) == 2:
                try:
                    idx = int(parts[0])
                    angle = int(parts[1])
                    cam_images[idx] = (angle, str(p))
                except ValueError:
                    continue

    if len(cam_images) >= 4:
        frames = []
        used_paths = set()
        for idx in sorted(cam_images.keys()):
            angle, path = cam_images[idx]
            dst_name = f'{idx:03d}_cam_{angle:03d}.png'
            frames.append((dst_name, angle, path, False))
            used_paths.add(path)

        # 保留未按 *_cam_* 命名的图片。它们通常是中车提供的真实/设计师
        # 视图；旧逻辑在已有 24 张渲染图时会静默丢弃这些图片。
        real_index = 0
        for p in sorted(map(str, image_paths)):
            if p not in used_paths:
                frames.append((f'real_{real_index:03d}.png', None, p, True))
                real_index += 1
        return frames

    # Pattern B: {idx:03d}.png (0, 1, 2, ..., N-1)
    numbered_images = {}
    for p in image_paths:
        stem = Path(p).stem
        try:
            idx = int(stem)
            numbered_images[idx] = str(p)
        except ValueError:
            continue

    if len(numbered_images) >= 4:
        frames = []
        for idx in sorted(numbered_images.keys()):
            angle = idx * 15
            dst_name = f'{idx:03d}_cam_{angle:03d}.png'
            frames.append((dst_name, angle, numbered_images[idx], False))
        return frames

    return None


def prepare_condition_images(image_paths, sha256, output_dir, image_size=1024):
    """
    复制条件图片到 renders_cond/{sha256}/，生成 transforms.json。

    自动检测单图 / 多图模式:
      - 单图 (无法解析多视角): 输出 000.png, transforms.json 含 1 帧
      - 多图: 输出 000_cam_000.png ~ 023_cam_345.png, transforms.json 含 24 帧

    支持 PNG, JPG, JPEG, WEBP 等格式
    """
    cond_dir = os.path.join(output_dir, 'renders_cond', sha256)
    os.makedirs(cond_dir, exist_ok=True)

    frames = _parse_multiview_pngs(image_paths)

    if frames is None:
        # ---- 单图模式 ----
        image_path = image_paths[0] if isinstance(image_paths, list) else image_paths
        img = Image.open(image_path).convert('RGBA')
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        img.save(os.path.join(cond_dir, '000.png'))

        transforms = {
            "camera_angle_x": 0.691111,
            "camera_angle_y": 0.691111,
            "frames": [{
                "file_path": "000.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }]
        }
        with open(os.path.join(cond_dir, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=2)

        print(f"  条件图片 (单图) → {cond_dir}/")
        return 1

    # ---- 多图模式 ----
    transform_frames = []
    real_count = 0
    for dst_name, angle_deg, src_path, is_real in frames:
        img = Image.open(src_path)
        img = img.convert('RGBA')  # 训练代码需要 RGBA (alpha 通道用于 bbox 裁剪)
        img.save(os.path.join(cond_dir, dst_name))

        frame = {
            "file_path": dst_name,
            "camera_angle_x": 0.523599,  # 30° FOV
            "is_real": is_real,
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
        if angle_deg is not None:
            frame["yaw_deg"] = angle_deg
        else:
            real_count += 1
        transform_frames.append(frame)

    transforms = {
        "camera_angle_x": 0.523599,
        "camera_angle_y": 0.523599,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "frames": transform_frames
    }
    with open(os.path.join(cond_dir, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)

    note = f"，其中真实图 {real_count} 张" if real_count else ""
    print(f"  条件图片 (多图, {len(frames)} 视角{note}) → {cond_dir}/")
    return len(frames)


def refresh_real_condition_images(image_paths, sha256, output_dir):
    """向已有 renders_cond 增量加入真实图片，不重写 24 张渲染视图。"""
    frames = _parse_multiview_pngs(image_paths)
    if not frames:
        return 0
    real_sources = [(name, src) for name, _, src, is_real in frames if is_real]
    if not real_sources:
        return 0

    cond_dir = os.path.join(output_dir, 'renders_cond', sha256)
    transforms_path = os.path.join(cond_dir, 'transforms.json')
    if not os.path.exists(transforms_path):
        return prepare_condition_images(image_paths, sha256, output_dir)

    with open(transforms_path) as f:
        transforms = json.load(f)
    existing = {frame['file_path'] for frame in transforms['frames']}
    added = 0
    for dst_name, src_path in real_sources:
        if dst_name in existing:
            continue
        Image.open(src_path).convert('RGBA').save(os.path.join(cond_dir, dst_name))
        transforms['frames'].append({
            'file_path': dst_name,
            'camera_angle_x': 0.523599,
            'is_real': True,
            'transform_matrix': [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })
        added += 1
    if added:
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        print(f"  增量加入 {added} 张真实条件图 → {cond_dir}/")
    return len(transforms['frames'])


def main():
    parser = argparse.ArgumentParser(description='自定义数据预处理 (无 Blender)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='原始数据目录 (含子目录，每目录有图片 + GLB/OBJ)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--resolutions', type=str, default='256,512,1024',
                        help='Dual grid 分辨率 (逗号分隔)')
    parser.add_argument('--flat', action='store_true',
                        help='Flat 模式: data_dir 下直接是 GLB/OBJ 文件 (非子目录结构)')
    parser.add_argument('--conditions-only', action='store_true',
                        help='仅刷新 renders_cond 和 metadata 中的条件图字段，不重建 dual grid / mesh dump')
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.resolutions = [int(x) for x in opt.resolutions.split(',')]

    if not opt.conditions_only:
        global o_voxel
        import o_voxel

    data_dir = opt.data_dir
    output_dir = opt.output_dir

    # 创建输出目录结构
    for res in opt.resolutions:
        os.makedirs(os.path.join(output_dir, f'dual_grid_{res}'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'renders_cond'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mesh_dumps'), exist_ok=True)

    # 扫描数据目录
    entries = []
    stats = {'glb+image': 0, 'glb_only': 0, 'skipped': 0}

    if opt.get('flat', False):
        # ================================================================
        # Flat 模式: data_dir 下直接是 3D 模型文件，图片按 stem 匹配
        # ================================================================
        print(f"Flat 模式: 扫描 {data_dir} 下的 3D 模型文件...")

        # 收集所有 3D 模型文件 (GLB, OBJ 等)
        mesh_files = []
        for ext in ['*.glb', '*.GLB', '*.obj', '*.OBJ']:
            mesh_files.extend(list(Path(data_dir).glob(ext)))

        if not mesh_files:
            print(f"[错误] 目录中没有 GLB/OBJ 文件: {data_dir}")
            sys.exit(1)

        # 建立图片索引: stem → 第一个匹配的图片路径
        image_index = {}
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.webp', '*.WEBP']:
            for img_path in Path(data_dir).glob(ext):
                stem = img_path.stem
                if stem not in image_index:
                    image_index[stem] = str(img_path)

        for mesh_path in sorted(mesh_files):
            stem = mesh_path.stem
            has_image = stem in image_index

            entry = {
                'name': stem,
                'mesh': str(mesh_path),
                'sha256': None if opt.conditions_only else sha256_file(str(mesh_path)),
                'has_image': has_image,
            }
            if has_image:
                entry['pngs'] = [image_index[stem]]
                stats['glb+image'] += 1
            else:
                stats['glb_only'] += 1

            entries.append(entry)

        print(f"  找到 {len(mesh_files)} 个模型文件, {len(image_index)} 张图片")
    else:
        # ================================================================
        # 子目录模式: 每个子目录包含 3D 模型 [+ 图片]
        # ================================================================
        for subdir in sorted(os.listdir(data_dir)):
            subpath = os.path.join(data_dir, subdir)
            if not os.path.isdir(subpath):
                continue

            # 查找图片 (支持多种格式)
            images = []
            for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.webp', '*.WEBP']:
                images.extend(list(Path(subpath).glob(ext)))

            # 查找 3D 模型 (GLB, OBJ 等)
            meshes = (list(Path(subpath).glob('*.glb')) + list(Path(subpath).glob('*.GLB')) +
                      list(Path(subpath).glob('*.obj')) + list(Path(subpath).glob('*.OBJ')))

            if not meshes:
                print(f"[跳过] {subdir}: 没有 GLB/OBJ 文件")
                stats['skipped'] += 1
                continue

            has_image = len(images) > 0
            entry = {
                'name': subdir,
                'mesh': str(meshes[0]),
                'sha256': None if opt.conditions_only else sha256_file(str(meshes[0])),
                'has_image': has_image,
            }
            if has_image:
                # 收集所有图片路径 (支持多视角)
                entry['pngs'] = [str(img) for img in sorted(images)]
                n_views = len(entry['pngs'])
                stats['glb+image'] += 1
                if n_views > 1:
                    stats.setdefault('multiview', 0)
                    stats['multiview'] += 1
            else:
                print(f"[注意] {subdir}: 没有图片文件，将跳过条件渲染")
                stats['glb_only'] += 1

            entries.append(entry)

        print(f"\n找到 {len(entries)} 条有效数据 (共扫描 {len(os.listdir(data_dir))} 个子目录)")

    print(f"  模型+图片: {stats['glb+image']}")
    if stats.get('multiview', 0) > 0:
        print(f"    其中多视角: {stats['multiview']}")
    print(f"  仅模型:    {stats['glb_only']}")
    print(f"  跳过:      {stats['skipped']}\n")

    # 条件图刷新不需要重读 107G 原始网格计算 SHA。复用 metadata 中的
    # local_path -> sha256 映射；只有新加入但尚未预处理的模型才回退到哈希。
    if opt.conditions_only:
        import pandas as pd
        metadata_path = os.path.join(output_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise ValueError('--conditions-only requires an existing metadata.csv')
        existing_metadata = pd.read_csv(metadata_path)
        sha_by_path = {
            str(Path(path).resolve()): sha
            for sha, path in zip(existing_metadata['sha256'], existing_metadata['local_path'])
        }
        for entry in entries:
            entry['sha256'] = sha_by_path.get(str(Path(entry['mesh']).resolve()))
            if entry['sha256'] is None:
                print(f"[注意] 新模型，计算 SHA: {entry['name']}")
                entry['sha256'] = sha256_file(entry['mesh'])

    # 处理每条数据
    records = []
    seen_sha256 = set()
    for i, entry in enumerate(entries):
        name = entry['name']
        sha256 = entry['sha256']
        mesh = entry['mesh']
        print(f"[{i+1}/{len(entries)}] {name} (sha256={sha256})")

        # 预处理产物以 sha256 为文件名。相同网格不能拥有两套不同的
        # renders_cond/latent；保留首次出现的样本，避免 metadata 产生重复 ID。
        if sha256 in seen_sha256:
            print(f"  [跳过] 与前一模型内容重复 (sha256={sha256})")
            continue
        seen_sha256.add(sha256)

        try:
            if opt.conditions_only:
                if not entry.get('has_image', False):
                    print(f"  [跳过] 无条件图片")
                    continue
                n_views = refresh_real_condition_images(entry['pngs'], sha256, output_dir)
                records.append({
                    'sha256': sha256,
                    'name': name,
                    'local_path': mesh,
                    'cond_rendered': True,
                    'num_views': n_views,
                })
                continue

            # Step 1: 3D 模型 → mesh
            print(f"  加载模型: {mesh}")
            vertices, faces = load_and_normalize_mesh(mesh)
            print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")

            # Step 2: Mesh → dual grid .vxz
            dual_results = convert_to_dual_grid(
                vertices, faces, sha256, output_dir, opt.resolutions
            )

            # Step 2b: 创建 mesh dump（Shape VAE 训练需要）
            create_mesh_dump(vertices, faces, sha256, output_dir)

            # Step 3: 准备条件图片 (自动单图/多图)
            if entry.get('has_image', False):
                n_views = prepare_condition_images(entry['pngs'], sha256, output_dir)
                cond_rendered = True
            else:
                print(f"  [跳过] 无条件图片，略过 renders_cond")
                cond_rendered = None
                n_views = 0

            records.append({
                'sha256': sha256,
                'name': name,
                'local_path': mesh,
                'cond_rendered': cond_rendered,
                'mesh_dumped': True,
                'pbr_dumped': False,
                'num_faces': len(faces),
                'num_views': n_views,
                'dual_grid_converted': True,  # encode_shape_latent.py 需要
                'dual_grid_size': dual_results.get(512, 0),
                **{f'dual_grid_converted_{r}': True for r in opt.resolutions},
                **{f'dual_grid_size_{r}': dual_results.get(r, 0) for r in opt.resolutions},
                'aesthetic_score': 5.5,
            })

        except Exception as e:
            print(f"  [错误] {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 4: 生成 metadata.csv
    if records:
        import pandas as pd
        csv_path = os.path.join(output_dir, 'metadata.csv')
        df = pd.DataFrame.from_records(records)
        if opt.conditions_only and os.path.exists(csv_path):
            # 保留现有的 dual-grid、mesh、latent 等字段，仅更新条件视图元数据。
            existing = pd.read_csv(csv_path).drop_duplicates('sha256', keep='first').set_index('sha256')
            refreshed = df.drop_duplicates('sha256', keep='first').set_index('sha256')
            existing.update(refreshed)
            missing = refreshed.index.difference(existing.index)
            if len(missing) > 0:
                existing = pd.concat([existing, refreshed.loc[missing]], axis=0)
            df = existing.reset_index()
        df.to_csv(csv_path, index=False)
        print(f"\n✓ metadata.csv 已保存到 {csv_path}")
        print(f"  共 {len(records)} 条记录")
        print(f"\n  有模型的数据: {len(records)}")
        has_glb_count = df['mesh_dumped'].sum() if 'mesh_dumped' in df.columns else 0
        print(f"  Dual grid 已转换: {sum(1 for r in records if all(f'dual_grid_converted_{res}' in r and r[f'dual_grid_converted_{res}'] for res in opt.resolutions))}")
    else:
        print("\n[警告] 没有成功处理任何数据!")


if __name__ == '__main__':
    main()
