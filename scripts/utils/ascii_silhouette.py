#!/usr/bin/env python3
"""把渲染图降采样成 ASCII 剪影，便于在无法传图时可靠核对形状。

用法: ascii_silhouette.py <渲染目录> [视角下标...]
"""
import sys
import glob
import os

import numpy as np
from PIL import Image

CHARS = ' .:-=+*#%@'


def silhouette(path, cols=104, rows=26):
    im = Image.open(path).convert('RGBA')
    a = np.array(im)[:, :, 3].astype(np.float32) / 255
    # 按字符宽高比 2:1 采样
    small = np.array(Image.fromarray((a * 255).astype(np.uint8))
                     .resize((cols, rows), Image.BOX)).astype(np.float32) / 255
    lines = []
    for r in range(rows):
        lines.append(''.join(CHARS[min(len(CHARS) - 1, int(v * len(CHARS)))]
                             for v in small[r]))
    return lines, float((a > 0.1).mean())


def gray_ascii(path, cols=104, rows=26):
    """物体区域用亮度字符表示（背景留空），用于看涂装/细节。"""
    im = Image.open(path).convert('RGBA')
    arr = np.array(im).astype(np.float32) / 255
    a = arr[:, :, 3]
    lum = arr[:, :, :3].mean(axis=2) * (a > 0.1)
    sm_a = np.array(Image.fromarray((a * 255).astype(np.uint8))
                    .resize((cols, rows), Image.BOX)).astype(np.float32) / 255
    sm_l = np.array(Image.fromarray((lum * 255).astype(np.uint8))
                    .resize((cols, rows), Image.BOX)).astype(np.float32) / 255
    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if sm_a[r, c] < 0.25:
                row.append(' ')
            else:
                row.append(CHARS[min(len(CHARS) - 1,
                                     max(1, int(sm_l[r, c] * len(CHARS))))])
        lines.append(''.join(row))
    return lines


def main():
    d = sys.argv[1]
    idxs = [int(x) for x in sys.argv[2:]] or [0, 6, 12, 18]
    pngs = sorted(glob.glob(os.path.join(d, '*_cam_*.png')))
    print(f'### {d}  （{len(pngs)} 张）')
    for i in idxs:
        if i >= len(pngs):
            continue
        lines, cov = silhouette(pngs[i])
        print(f'\n--- {os.path.basename(pngs[i])}  alpha覆盖={cov:.4f} ---')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()