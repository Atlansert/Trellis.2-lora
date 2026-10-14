#!/usr/bin/env python3
"""对指定的 SS LoRA 检查点，在 data_test 上做**多种子**朝向评测（SS 层，秒级）。

为什么在 SS 层评测：完整推理要过 shape/tex/remesh（分钟级，003 的 remesh 曾卡 100 分钟），
而"翻转"完全由 SS 输出的长度轴决定，SS 层即可判定。

判据（规范化后的数据已对齐到管线约定）：
  正确 ⇔ SS 占据体长轴 = index1（成品 GLB 长度轴 = z）。
"""
import argparse
import os
import sys
from collections import Counter

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/data/lzx/TRELLIS.2')
sys.path.insert(0, '/data/lzx/TRELLIS.2')

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.lora import apply_lora_to_linear_layers

TARGETS = ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
           'cross_attn.to_kv', 'cross_attn.to_out']
P = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
         guidance_interval=[0.6, 1.0], rescale_t=5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='tag=path ...')
    ap.add_argument('--images', nargs='+', default=['001', '002', '003', '004'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47, 48, 49])
    ap.add_argument('--canonicalize', action='store_true',
                    help='同时开启逐样本最长轴规范化（仅作对照）')
    a = ap.parse_args()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('checkpoints')
    pipe.cuda()
    ss = pipe.models['sparse_structure_flow_model']
    if a.canonicalize:
        pipe.ss_canonicalize = True

    imgs = {n: pipe.preprocess_image(Image.open(f'data_test/{n}.png')) for n in a.images}
    conds = {n: pipe.get_ss_cond([imgs[n]]) for n in a.images}

    def occ_axis(cond, seed):
        """返回 (argmax 长轴, E1/E2 比值)。

        只用 argmax 会在"斜视"输出上失效：物体转到 ~45° 时 index1 与 index2 的
        跨度几乎相等（都 ≈0.707L），argmax 变成掷硬币，看起来像翻转其实不是。
        因此判据用 **ratio = E1 / max(E0, E2)**：
          ratio > 1.5  → 长度确实落在水平轴（正确）
          ratio < 0.67 → 长度落在竖直或视深轴（明确错误，含"立起来"的俯仰错）
          其余          → 斜视/欠定
        E1/E2 单独看会漏掉"俯仰错"（竖直轴最长时 E1/E2 仍可能 >1）。"""
        torch.manual_seed(seed)
        c = pipe.sample_sparse_structure(cond, 64, 1, dict(P)).cpu().numpy()
        # ★ 忠实模拟管线：--canonicalize 的置换是在 pipeline.run() 里做的，
        # 而本脚本直接调 sample_sparse_structure，故必须在此复刻，否则开关无效。
        if a.canonicalize and len(c):
            cc = c.copy()
            for b in range(int(c[:, 0].max().item()) + 1):
                m = c[:, 0] == b
                if int(m.sum()) == 0:
                    continue
                sp = c[m][:, 1:].astype(np.float32)
                la = int(np.argmax(sp.max(0) - sp.min(0) + 1))
                if la != 1:
                    perm = [0, 1, 2, 3]
                    perm[1 + la], perm[2] = perm[2], perm[1 + la]
                    cc[m] = c[m][:, perm]
            c = cc
        o = np.zeros((64, 64, 64), bool)
        for d, h, w in c[:, 1:]:
            o[d, h, w] = True
        if not len(c):
            return '-', float('nan')
        nz = np.argwhere(o)
        e = nz.max(0) - nz.min(0) + 1
        return 'xyz'[int(np.argmax(e))], float(e[1] / max(e[0], e[2], 1))

    # ★ LoRA 只能在循环外注入一次：若每个检查点都注入，会层层叠加
    # （参数名变成 lora_A.lora_A...），表现为 missing 键数线性增长、结果全错。
    apply_lora_to_linear_layers(ss, TARGETS, rank=8, alpha=8.0, dropout=0.0)
    for spec in a.ckpts:
        tag, path = spec.split('=', 1)
        if os.path.isfile(path):
            sd = torch.load(path, map_location='cpu', weights_only=True)
            sd.pop('rope_phases', None)
            r = ss.load_state_dict(sd, strict=False)
            miss = [k for k in r.missing_keys if k != 'rope_phases']
            print(f'\n=== {tag}  (missing={len(miss)}) ===', flush=True)
        else:
            print(f'\n=== {tag} <缺失> ===')
            continue
        allok = 0
        tot = 0
        for n in a.images:
            res = [occ_axis(conds[n], s) for s in a.seeds]
            axs = [r[0] for r in res]
            ratios = np.array([r[1] for r in res], dtype=float)
            cnt = Counter(axs)
            # 判据（二选一都算"明确正确"）：长轴=index1，或 E1/E2>1.5
            ok = sum(1 for ax, rr in res if rr == rr and rr > 1.5)
            allok += ok
            tot += len(res)
            dist = ' '.join(f'{k}:{100 * v / len(res):.0f}%' for k, v in cnt.most_common())
            med = np.nanmedian(ratios)
            amb = int(np.sum((ratios >= 0.67) & (ratios <= 1.5)))
            print(f'  {n}  {dist:<22} E1/max(E0,E2) 中位={med:>5.2f} 斜视={amb}/{len(res)}'
                  f'  明确正确 {ok}/{len(res)}', flush=True)
            print(f'      {axs}  比值 {np.round(ratios, 2).tolist()}', flush=True)
        print(f'  --> 明确正确率 {allok}/{tot} = {100 * allok / max(tot, 1):.0f}%', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())