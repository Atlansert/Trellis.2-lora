#!/usr/bin/env python3
"""把两组渲染图拼成上下对照的小 JPEG（用于带回本地目视核对）。

布局：上排 = A 组，下排 = B 组；每排取 --idx 指定的视角。
CLI: --a 上排目录 --b 下排目录 --out 输出.jpg [--idx 0,6] [--tile 256] [--quality 68]
安全边界：只读两个目录的 *_cam_*.png，只写 --out 一个文件；任一组无图直接报错退出。
"""
import os
import sys
import glob
import argparse

from PIL import Image

BG = (22, 22, 26)


def sheet(src, idxs, tile):
    """取 src 下按序号 idxs 的渲染图，合成深色底、缩放到 tile，返回图像列表（越界序号自动忽略）。"""
    pngs = sorted(glob.glob(os.path.join(src, '*_cam_*.png')))
    if not pngs:
        raise SystemExit(f'{src}: 无渲染图')
    out = []
    for i in idxs:
        if i >= len(pngs):
            continue
        im = Image.open(pngs[i]).convert('RGBA')
        base = Image.new('RGBA', im.size, BG + (255,))
        im = Image.alpha_composite(base, im).convert('RGB')
        out.append(im.resize((tile, tile), Image.LANCZOS))
    return out


def main():
    """拼上下对照图：上排 A 组、下排 B 组，列数取两组中较多者。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='上排渲染目录')
    ap.add_argument('--b', required=True, help='下排渲染目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--idx', default='0,6')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--quality', type=int, default=68)
    a = ap.parse_args()
    idxs = [int(x) for x in a.idx.split(',')]
    ra, rb = sheet(a.a, idxs, a.tile), sheet(a.b, idxs, a.tile)
    cols = max(len(ra), len(rb))
    canvas = Image.new('RGB', (a.tile * cols, a.tile * 2), BG)
    for k, im in enumerate(ra):
        canvas.paste(im, (k * a.tile, 0))
    for k, im in enumerate(rb):
        canvas.paste(im, (k * a.tile, a.tile))
    canvas.save(a.out, 'JPEG', quality=a.quality, optimize=True)
    print(f'{a.out}: {canvas.size} {os.path.getsize(a.out)/1024:.1f} KB')
    return 0


if __name__ == '__main__':
    sys.exit(main())