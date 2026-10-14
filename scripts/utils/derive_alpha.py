#!/usr/bin/env python3
"""从不透明条件图推导 alpha 通道，生成"透明背景"版本（与推理 preprocess_image 同口径）。

为什么不重渲：data_for_train 里 327 个 .obj 的 .mtl **全部缺失**（0/327），
重渲会得到无贴图的灰模，反而比存档更差；而存档渲染本身已带正确贴图/相机/色调，
只差一个 alpha 通道。本脚本精确补上这个通道：

  1. 背景是均匀世界背景 → 用边框环取中位色作为 bg
  2. mask = |RGB - bg| > tol
  3. 形态学闭运算 + 取最大连通域 + 填洞（避免车体上的白色区域被打成孔洞）
  4. alpha = mask*255，RGB 原样保留（训练/推理侧会 x*alpha 把背景变黑）

--test 模式只体检若干对象，不写文件。

CLI: [--test N 只体检] [--tol 3.0 背景色差阈值] [--apply] [--out renders_cond_t] [--limit N]
输出：默认模式把补好 alpha 的 RGBA 图与 transforms.json 写到 <ROOT>/<out>/<sha>/（会新建/覆盖）。
安全边界：--test 只打印，不写文件；默认模式只写输出目录，renders_cond 原图只读；只处理"不透明"对象。
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = 'training_data/multiview_1186'
CH = ' .:-=+*#%@'


def border_bg(arr):
    """取图像四边窄条的中位色作为背景估计，并给出环内标准差最大值（背景是否均匀）。"""
    h, w = arr.shape[:2]
    k = max(2, min(h, w) // 64)
    ring = np.concatenate([
        arr[:k].reshape(-1, 3), arr[-k:].reshape(-1, 3),
        arr[:, :k].reshape(-1, 3), arr[:, -k:].reshape(-1, 3)])
    return np.median(ring, axis=0), ring.std(axis=0).max()


def derive_mask(rgb, tol):
    """由 RGB 推前景掩膜：与背景色差 > tol → 闭运算 → 只留最大连通域 → 填洞。
    返回 (原掩膜, 填洞后掩膜, 背景色, 背景std, 色差图)。
    """
    bg, bgstd = border_bg(rgb)
    diff = np.abs(rgb - bg).max(axis=2)
    m = diff > tol
    # 闭运算桥接细小缝隙
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)), iterations=1)
    # 取最大连通域
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    # 填洞
    filled = ndimage.binary_fill_holes(m)
    return m, filled, bg, bgstd, diff


def show(m, cols=72, rows=18):
    """把布尔掩膜降采样成 ASCII 预览（便于在无图环境核对形状）。"""
    small = np.array(Image.fromarray((m * 255).astype(np.uint8))
                     .resize((cols, rows), Image.BOX)).astype(np.float32) / 255
    for r in range(rows):
        print('      |' + ''.join(CH[min(9, int(v * 10))] for v in small[r]) + '|')


def opaque_rows():
    """从 metadata.csv 挑出"不透明"对象（首帧 alpha 覆盖率 ≥0.99）的元数据行。"""
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'metadata.csv'))))
    out = []
    for r in rows:
        p = os.path.join(ROOT, 'renders_cond', r['sha256'], '000_cam_000.png')
        if os.path.isfile(p):
            a = np.array(Image.open(p).convert('RGBA'))[:, :, 3]
            if (a > 0).mean() >= 0.99:
                out.append(r)
    return out


def main():
    """--test 时打印若干对象的掩膜体检表；默认模式逐对象逐帧补 alpha 并写出新目录。"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, default=0, help='只体检 N 个对象')
    ap.add_argument('--tol', type=float, default=3.0, help='背景差异阈值（0-255）')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--out', default='renders_cond_t')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rows = opaque_rows()
    if a.limit:
        rows = rows[:a.limit]
    print(f'不透明对象 {len(rows)} 个，tol={a.tol}')

    if a.test:
        picks = [rows[i] for i in np.linspace(0, len(rows) - 1, a.test).astype(int)]
        print(f'{"对象":<14}{"来源":>6}{"背景色":>22}{"背景std":>9}'
              f'{"mask覆盖率":>11}{"填洞后":>9}{"填充比":>8}{"洞像素":>8}')
        for r in picks:
            p = os.path.join(ROOT, 'renders_cond', r['sha256'], '000_cam_000.png')
            arr = np.array(Image.open(p).convert('RGB')).astype(np.float32)
            m, filled, bg, bgstd, diff = derive_mask(arr, a.tol)
            nz = np.argwhere(m)
            ext = (nz.max(0) - nz.min(0) + 1) if len(nz) else (0, 0)
            fill = max(ext) / arr.shape[0] if len(nz) else 0.0
            holes = int(filled.sum() - m.sum())
            print(f'{r["sha256"][:12]:<14}{os.path.splitext(r["local_path"])[1]:>6}'
                  f'{str(np.round(bg,1)):>22}{bgstd:>9.3f}{m.mean():>11.4f}'
                  f'{filled.mean():>9.4f}{fill:>8.3f}{holes:>8d}')
        print('\n掩膜形状示例（填洞后，yaw=0）：')
        r = picks[0]
        p = os.path.join(ROOT, 'renders_cond', r['sha256'], '000_cam_000.png')
        arr = np.array(Image.open(p).convert('RGB')).astype(np.float32)
        _, filled, _, _, _ = derive_mask(arr, a.tol)
        print(f'  {r["sha256"][:12]}')
        show(filled)
        return 0

    outroot = os.path.join(ROOT, a.out)
    os.makedirs(outroot, exist_ok=True)
    n = 0
    for r in rows:
        src = os.path.join(ROOT, 'renders_cond', r['sha256'])
        dst = os.path.join(outroot, r['sha256'])
        os.makedirs(dst, exist_ok=True)
        tf = json.load(open(os.path.join(src, 'transforms.json')))
        for f in tf['frames']:
            sp = os.path.join(src, f['file_path'])
            if not os.path.isfile(sp):
                continue
            dp = os.path.join(dst, f['file_path'])
            im = Image.open(sp)
            if 'yaw_deg' in f:                      # 渲染帧：补 alpha
                arr = np.array(im.convert('RGB')).astype(np.float32)
                _, filled, _, _, _ = derive_mask(arr, a.tol)
                out = np.dstack([arr, filled.astype(np.float32) * 255])
                Image.fromarray(out.astype(np.uint8), 'RGBA').save(dp)
            else:                                    # real_*.png：原样拷贝
                im.convert('RGBA').save(dp)
        json.dump(tf, open(os.path.join(dst, 'transforms.json'), 'w'), indent=2)
        n += 1
        if n % 50 == 0:
            print(f'  {n}/{len(rows)}', flush=True)
    print(f'完成 {n} 个 → {outroot}')
    return 0


if __name__ == '__main__':
    sys.exit(main())