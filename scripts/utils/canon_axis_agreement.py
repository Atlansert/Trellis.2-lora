#!/usr/bin/env python3
"""判决：规范化后的 GT（施加轴置换 (2,0,1) 之后）是否落在原厂 base 的坐标系里。

判据：同一对象/视角，base（不挂 LoRA）在对应渲染图上预测的占据体长轴，
应当与 GT 体素（置换后）的长轴一致。base 本身有多种子双稳态，故取多种子多数票。

一致率高 ⇒ 数据约定正确，训练出的模型会直接输出管线约定（无需任何推理侧开关）。

运行：无参数（N_OBJ/VIEWS/SEEDS 为文件内常量）；需 GPU（默认卡 5），脚本内已 chdir 到项目根。
安全边界：只读 GT npz、renders_cond 图与 base 权重，不写任何文件；纯判定实验。
"""
import os
import sys
import glob
from collections import Counter

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline

P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
         guidance_interval=[0.6, 1.0], rescale_t=5.0)
ROOT = 'training_data/multiview_1186'
VOX = 'ss_view_voxels_canon_v4'
PERM = (2, 0, 1)          # 编码时施加的置换
N_OBJ = 24
VIEWS = [0, 3, 6, 9, 12, 15, 18, 21]
SEEDS = [42, 43, 44]


def main():
    """对每个对象/视角：base 多种子采样取长轴多数票，与 GT(置换后)长轴比较，输出逐项一致率。"""
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()

    def base_axis(cond):
        """对多个种子各采样一次 SS，取"最长轴"的多数票（base 朝向双稳态，多数票更稳）。"""
        ax = []
        for s in SEEDS:
            torch.manual_seed(s)
            c = pipe.sample_sparse_structure(cond, 64, 1, dict(P)).cpu().numpy()
            if not len(c):
                continue
            o = np.zeros((64, 64, 64), bool)
            for d, h, w in c[:, 1:]:
                o[d, h, w] = True
            nz = np.argwhere(o)
            ax.append('xyz'[int(np.argmax(nz.max(0) - nz.min(0) + 1))])
        return Counter(ax).most_common(1)[0][0] if ax else '-'

    def gt_axis(path):
        """读 GT 体素，先施加编码时的轴置换 PERM，再返回最长跨度所在轴；空体素返回 '-'。"""
        c = np.load(path)['coords'].astype(np.int64)
        if not len(c):
            return '-'
        c = c[:, PERM]
        nz = c
        return 'xyz'[int(np.argmax(nz.max(0) - nz.min(0) + 1))]

    objs = sorted(os.listdir(os.path.join(ROOT, VOX)))[:N_OBJ]
    agree = 0
    tot = 0
    rows = []
    for o in objs:
        rp = sorted(glob.glob(os.path.join(ROOT, 'renders_cond', o, '*.png')))
        for v in VIEWS:
            gp = os.path.join(ROOT, VOX, o, f'{v:03d}.npz')
            if not os.path.isfile(gp) or v >= len(rp):
                continue
            img = pipe.preprocess_image(Image.open(rp[v]))
            ba = base_axis(pipe.get_ss_cond([img]))
            ga = gt_axis(gp)
            ok = ba == ga
            agree += ok
            tot += 1
            rows.append((o, v, ba, ga, ok))
        print(f'{o}: {tot} 对, 一致 {agree}', flush=True)

    print(f'\n样本对数 = {tot}')
    print(f'base 长轴 与 GT(置换后) 长轴 一致率 = {agree}/{tot} = {100 * agree / max(tot, 1):.0f}%')
    print(f'\n{"对象":<20}{"view":>5}{"base":>6}{"GT":>5}  一致')
    for o, v, ba, ga, ok in rows[:40]:
        print(f'{o:<20}{v:>5}{ba:>6}{ga:>5}  {"OK" if ok else "xx"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())