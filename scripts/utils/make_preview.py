"""把一组渲染图拼成小尺寸 JPEG 预览图（用于把远端结果带回本地目视检查）。"""
import sys
import glob
import argparse

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='渲染图目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--idx', default='0,6,12,18')
    ap.add_argument('--tile', type=int, default=384)
    ap.add_argument('--quality', type=int, default=72)
    a = ap.parse_args()

    idxs = [int(x) for x in a.idx.split(',')]
    pngs = sorted(glob.glob(f'{a.src}/*_cam_*.png'))
    if not pngs:
        print('无渲染图'); return 1
    picks = [pngs[i] for i in idxs if i < len(pngs)]
    n = len(picks)
    cols = 2
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (a.tile * cols, a.tile * rows), (24, 24, 28))
    for k, p in enumerate(picks):
        im = Image.open(p).convert('RGBA')
        # 透明背景合成到深色底，便于看清轮廓
        bg = Image.new('RGBA', im.size, (24, 24, 28, 255))
        im = Image.alpha_composite(bg, im).convert('RGB')
        im = im.resize((a.tile, a.tile), Image.LANCZOS)
        canvas.paste(im, ((k % cols) * a.tile, (k // cols) * a.tile))
    canvas.save(a.out, 'JPEG', quality=a.quality, optimize=True)
    import os
    print(f'{a.out}: {canvas.size} {os.path.getsize(a.out)/1024:.1f} KB')
    return 0


if __name__ == '__main__':
    sys.exit(main())