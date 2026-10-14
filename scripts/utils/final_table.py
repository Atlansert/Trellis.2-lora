#!/usr/bin/env python3
"""最终交付表：任务 4 扫描（A 组）与 yz 修正组的朝向 + 几何指标。"""
import os
import re
import glob

import numpy as np
import trimesh


def metrics(p):
    import subprocess
    out = subprocess.run(
        ['/home/test/miniconda3/envs/trellis2/bin/python',
         'scripts/utils/mesh_metrics.py', p],
        capture_output=True, text=True, cwd='/data/lzx/TRELLIS.2').stdout
    for ln in out.splitlines():
        if '.glb' in ln and '完成' not in ln and '实验' not in ln:
            f = ln.split()
            # 文件 顶点 面 尖边占比 夹角均值 夹角p95 边界边占比 分量 封闭
            try:
                return float(f[-6]), float(f[-5]), float(f[-4])
            except Exception:
                return None, None, None
    return None, None, None


def row(label, p):
    m = trimesh.load(p, force='mesh', process=False, skip_materials=True)
    b = np.asarray(m.bounds[1] - m.bounds[0])
    ax = 'xyz'[int(np.argmax(b))]
    sf, dm, p95 = metrics(p)
    return label, ax, dm, p95, sf


def main():
    print('=' * 96)
    print('任务 4 交付：SS LoRA 10k–50k（shape flow 固定 20k，data_test/001.png）')
    print('=' * 96)
    print(f'{"配置":<28}{"步数":>8}{"长轴":>6}{"朝向":>10}'
          f'{"夹角均值↓":>11}{"夹角p95↓":>10}{"尖边占比↓":>11}')
    rows = []
    for s, tag in [('0010000', '10k'), ('0020000', '20k'), ('0030000', '30k'),
                   ('0040000', '40k'), ('0050000', '50k')]:
        p = f'results_infer/sweep_ss_dinov3_{s}_shape20k/001.glb'
        if os.path.isfile(p) and os.path.getsize(p):
            rows.append(('A 视角帧（不加 y/z 交换）', tag, p))
        q = glob.glob(f'results_infer/yzfix_ssA{"0"+s[1:] if s!="0020000" else "20k"}_shape20k/001.glb')
        if s == '0020000':
            q = ['results_infer/yzfix_ssA20k_shape20k/001.glb']
        if q and os.path.isfile(q[0]) and os.path.getsize(q[0]):
            rows.append(('A + y/z 交换（可用）', tag, q[0]))
    base = 'results_infer/data_test_stage1_lora_33k_stage2_lora_20k/001.glb'
    if os.path.isfile(base):
        rows.append(('线上旧基线 33k（参照）', '—', base))
    for label, tag, p in rows:
        try:
            lb, ax, dm, p95, sf = row(label, p)
            dm_s = f'{dm:.3f}' if dm is not None else '—'
            p95_s = f'{p95:.2f}' if p95 is not None else '—'
            sf_s = f'{sf:.4f}' if sf is not None else '—'
            print(f'{lb:<28}{tag:>8}{ax:>6}'
                  f'{("✅规范" if ax == "z" else "❌转90°"):>10}'
                  f'{dm_s:>11}{p95_s:>10}{sf_s:>11}')
        except Exception as e:
            print(f'{label:<28}{tag:>8}  读取失败: {e}')
    print()
    print('朝向判据：管线期望 SS 输出为"已对齐"坐标系（成品长度轴 = z）。')
    print('夹角均值/p95/尖边占比 越小越平滑；但退化几何也会"更平滑"，须与朝向联读。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())