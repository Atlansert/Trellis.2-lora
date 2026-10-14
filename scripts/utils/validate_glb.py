#!/usr/bin/env python3
"""校验暂存区 GLB 是否为有效且非空的 glTF 二进制（含 mesh）。"""
import os
import sys
import json
import struct
import argparse


def probe_glb(path):
    """返回 (ok, reason, info)。"""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(12)
        if len(head) < 12:
            return False, 'file too small', {}
        magic, version, length = struct.unpack('<4sII', head)
        if magic != b'glTF':
            return False, f'bad magic {magic!r}', {}
        if length != size:
            return False, f'declared length {length} != file size {size}', {}
        clen, ctype = struct.unpack('<I4s', f.read(8))
        if ctype != b'JSON':
            return False, f'first chunk not JSON ({ctype!r})', {}
        doc = json.loads(f.read(clen).decode('utf-8'))
    meshes = doc.get('meshes') or []
    prims = sum(len(m.get('primitives') or []) for m in meshes)
    if not meshes or prims == 0:
        return False, 'no mesh primitives', {'meshes': len(meshes)}
    if not doc.get('accessors'):
        return False, 'no accessors', {}
    info = {
        'meshes': len(meshes),
        'primitives': prims,
        'materials': len(doc.get('materials') or []),
        'images': len(doc.get('images') or []),
        'nodes': len(doc.get('nodes') or []),
    }
    return True, '', info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/lzx/TRELLIS.2/new_data_staged')
    a = ap.parse_args()
    units = sorted(d for d in os.listdir(a.root)
                   if os.path.isdir(os.path.join(a.root, d)))
    ok, bad = [], []
    for u in units:
        p = os.path.join(a.root, u, 'mesh_textured_pbr.glb')
        if not os.path.exists(p):
            bad.append((u, 'missing glb', {}))
            continue
        good, reason, info = probe_glb(p)
        (ok if good else bad).append((u, reason, info))
    print(f'有效: {len(ok)} / 共 {len(units)}')
    if bad:
        print(f'\n无效 {len(bad)} 个：')
        for u, r, i in bad:
            print(f'  ✗ {u}: {r}')
    # 无贴图（无 materials/images）的模型提示一下
    notex = [u for u, _, i in ok if i.get('images', 0) == 0 or i.get('materials', 0) == 0]
    print(f'\n无贴图/materials 的模型: {len(notex)}')
    for u in notex[:15]:
        print(f'  ! {u}')
    if len(notex) > 15:
        print(f'  ... 另 {len(notex)-15} 个')
    with open(os.path.join(a.root, '_valid_units.txt'), 'w') as f:
        for u, _, _ in ok:
            f.write(u + '\n')
    print(f'\n有效清单: {os.path.join(a.root, "_valid_units.txt")}')


if __name__ == '__main__':
    main()