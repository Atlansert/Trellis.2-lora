#!/usr/bin/env python3
"""判定修正后 GT 的手性（是否被镜像）。

y/z 对换是一个**对换**（行列式 -1），会把对象镜像；因此必须确认网格相对渲染图
是否左右颠倒。做法：把 GT 占据体沿"视深"轴投影成 (水平, 竖直) 二维剪影，
与渲染图 alpha 剪影（按前景 bbox 归一化到同尺寸）比较 IoU；再与**左右翻转**后的
GT 剪影比较。若翻转版明显更高，说明 GT 被镜像，需要补一次单轴取反。

网格轴语义（实测）：index0 = 画面水平、index1 = 视深、index2 = 画面竖直。

运行：无参数（N_OBJ/VIEWS 为文件内常量）；脚本内已 chdir 到项目根。
安全边界：只读 GT npz 与 renders_cond_t 图，不写任何文件；纯 CPU，不需 GPU。
"""
import os
import sys
import glob

os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
from PIL import Image

ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon'
N_OBJ = 60
VIEWS = [0, 4, 8, 12, 16, 20]
R = 64


def crop_norm(mask, res=R):
    """按前景 bbox 裁切并缩放到 res×res 的布尔剪影（统一尺寸后才能比 IoU）；前景太少返回全空。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return np.zeros((res, res), bool)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((res, res), Image.BOX)) > 127


def main():
    """对若干对象/视角：GT 投影剪影与渲染 alpha 剪影比 IoU，并与其左右翻转版比较，判定 GT 是否被镜像。"""
    objs = sorted(os.listdir(f'{ROOT}/{VOX}'))[:N_OBJ]
    ious, ious_flip = [], []
    for o in objs:
        rp = sorted(glob.glob(f'{ROOT}/renders_cond_t/{o}/*.png'))
        for v in VIEWS:
            gp = f'{ROOT}/{VOX}/{o}/{v:03d}.npz'
            if not os.path.isfile(gp) or v >= len(rp):
                continue
            c = np.load(gp)['coords'].astype(np.int64)
            g = np.zeros((R, R, R), bool)
            g[np.clip(c[:, 0], 0, R - 1), np.clip(c[:, 1], 0, R - 1), np.clip(c[:, 2], 0, R - 1)] = True
            # 沿视深(index1)投影 -> (水平=index0, 竖直=index2) 的二维剪影
            proj = g.any(axis=1)                     # (index0, index2)
            a = np.array(Image.open(rp[v]).convert('RGBA').getchannel('A'))
            rmask = a > 128
            if proj.sum() < 4 or rmask.sum() < 4:
                continue
            P1 = crop_norm(proj)
            P2 = crop_norm(np.fliplr(proj))
            # 渲染剪影：竖直方向与 index2 一致，水平与 index0 一致
            Rm = crop_norm(rmask.T)
            for arr, acc in ((P1, ious), (P2, ious_flip)):
                u = (arr | Rm).sum()
                acc.append(float((arr & Rm).sum()) / u if u else 0.0)
    ious = np.array(ious)
    ious_flip = np.array(ious_flip)
    print(f'样本数 = {len(ious)}')
    print(f'GT 剪影 vs 渲染剪影    平均IoU = {ious.mean():.3f}  中位 = {np.median(ious):.3f}')
    print(f'GT 左右翻转 vs 渲染    平均IoU = {ious_flip.mean():.3f}  中位 = {np.median(ious_flip):.3f}')
    d = ious - ious_flip
    print(f'\n逐样本差值 均值 = {d.mean():+.3f}  为正的比例 = {100 * (d > 0).mean():.0f}%')
    if ious_flip.mean() > ious.mean() + 0.01:
        print('=> GT 被镜像，需要补一次单轴取反')
    elif ious.mean() > ious_flip.mean() + 0.01:
        print('=> 手性正确，不需要取反')
    else:
        print('=> 无法判定（差异不显著）')
    return 0


if __name__ == '__main__':
    sys.exit(main())