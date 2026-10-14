#!/usr/bin/env python3
"""把 new_data 里"保留"的 GLB 资产整理成 data_for_train 的单元格式（暂存区）。

new_data 各交付包的结构约定：
    <Package>/delete|Deleted/   被否决的资产（不入册）
    <Package>/blender|Blender/  源工程 .blend（归档保留，不参与训练）
    <Package>/PLB|bip/          已导出的贴图 GLB（训练用模型）

输出（暂存区，渲染校验通过后再移入 data_for_train）：
    new_data_staged/<stem>/
        mesh_textured_pbr.glb     GLB 模型（统一命名，沿用 data_for_train 约定）
        source.json               来源溯源信息
"""
import os
import re
import json
import shutil
import hashlib
import argparse

PKGS = {
    'CAPTAIN_3D Data_Modification_Package10（Jiang）': 'PLB',
    'CAPTAIN_3D Data_Modification_Package13_701-800（Jiang）': 'bip',
    'CAPTAIN_3D Data_Modification_Package13_801-900（Jiang）': 'bip',
}
# 仅含 delete/ 的包：无保留资产
PKGS_REJECT_ONLY = ['CAPTAIN_3D Data_Modification_Package08_201-300（Jiang）']


def sha256_file(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(nbytes), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/lzx/TRELLIS.2')
    ap.add_argument('--staged', default='new_data_staged')
    ap.add_argument('--move', action='store_true',
                    help='直接移动而不是复制')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    staged = os.path.join(root, a.staged)
    os.makedirs(staged, exist_ok=True)

    picked = []
    for pkg, sub in PKGS.items():
        d = os.path.join(root, 'new_data', pkg, sub)
        if not os.path.isdir(d):
            print(f'[警告] 缺目录: {d}')
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith('.glb'):
                continue
            stem = os.path.splitext(f)[0]
            src = os.path.join(d, f)
            dst_dir = os.path.join(staged, stem)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, 'mesh_textured_pbr.glb')
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                print(f'[跳过] {stem} 已暂存')
                picked.append((pkg, sub, src, stem))
                continue
            if a.move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
            with open(os.path.join(dst_dir, 'source.json'), 'w') as fh:
                json.dump({
                    'package': pkg,
                    'package_subdir': sub,
                    'original_basename': f,
                    'original_path': src,
                    'sha256': sha256_file(src if not a.move else dst),
                    'concept_id': (re.match(r'(Concept\d+)', stem).group(1)
                                   if re.match(r'(Concept\d+)', stem) else None),
                }, fh, ensure_ascii=False, indent=2)
            picked.append((pkg, sub, src, stem))
            print(f'[暂存] {stem}  ({os.path.getsize(dst)/1e6:.1f} MB)')

    print(f'\n共暂存 {len(picked)} 个资产 → {staged}')
    for p in PKGS_REJECT_ONLY:
        d = os.path.join(root, 'new_data', p)
        if os.path.isdir(d):
            subs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            print(f'[包 {p}] 仅含 {subs} → 全部为否决资产，不入册')
    with open(os.path.join(staged, '_manifest.csv'), 'w') as fh:
        fh.write('stem,package,package_subdir,original_path\n')
        for pkg, sub, src, stem in picked:
            fh.write(f'{stem},{pkg},{sub},{src}\n')
    print(f'清单: {os.path.join(staged, "_manifest.csv")}')


if __name__ == '__main__':
    main()