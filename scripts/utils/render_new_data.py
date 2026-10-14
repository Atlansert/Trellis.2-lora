#!/usr/bin/env python3
"""批量渲染 24 视角条件图（Blender 4.5.1 + Cycles GPU + Filmic）。

与仓库自带 scripts/utils/render_views.py 的区别：
  * 使用本地 Blender 4.5.1（/data/lzx/opt），不做 sudo/下载
  * 每个 worker 绑定一张物理 GPU（CUDA_VISIBLE_DEVICES + --gpu 0）
  * --transparent：与官方 render_cond.py 及推理 preprocess_image 约定一致
  * 色彩管理由 render_view.py 钉死为 Filmic（与 data_for_train 存档一致）
  * --overwrite 强制重渲；逐单元日志；失败重试；断点续跑

用法：
  python scripts/utils/render_new_data.py --root new_data_staged \
      --gpus 5,6,7 --workers-per-gpu 2 --transparent

输出：在各单元目录写 24 张视角图 + transforms.json，日志写 <root>/_logs/<unit>.log，
      失败清单写 <root>/_failed.txt；退出码非 0 表示有失败单元。
安全边界：只读网格，会覆盖/新增各单元目录内的渲染产物（--overwrite 强制重渲）；用本地 Blender，不 sudo、不下载。
"""
import os
import sys
import glob
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

REPO = '/data/lzx/TRELLIS.2'
BLENDER = '/data/lzx/opt/blender-4.5.1-linux-x64/blender'
BSCRIPT = os.path.join(REPO, 'data_toolkit', 'blender_script', 'render_view.py')
LABELS = [f'cam_{y:03d}' for y in range(0, 360, 15)]


def is_done(unit_dir):
    """完成判据：transforms.json 存在且 24 张视角图齐全（用于断点续跑）。"""
    if not os.path.exists(os.path.join(unit_dir, 'transforms.json')):
        return False
    return all(os.path.exists(os.path.join(unit_dir, f'{i:03d}_{lab}.png'))
               for i, lab in enumerate(LABELS))


def find_mesh(unit_dir):
    """在单元目录里找第一个 glb/obj（不区分大小写扩展名），找不到返回 None。"""
    for ext in ('*.glb', '*.GLB', '*.obj', '*.OBJ'):
        g = sorted(glob.glob(os.path.join(unit_dir, ext)))
        if g:
            return g[0]
    return None


def render_one(unit, root, gpu, resolution, transparent, logdir, timeout, overwrite):
    """渲染单个单元：已完成的跳过（除非 overwrite），Blender 以 -b 后台跑 render_view.py。
    返回 (单元名, 状态, 耗时秒)；状态含 ok/skipped/no-mesh/failed/timeout/error。
    """
    unit_dir = os.path.join(root, unit)
    mesh = find_mesh(unit_dir)
    if mesh is None:
        return unit, 'no-mesh', 0.0
    if not overwrite and is_done(unit_dir):
        return unit, 'skipped', 0.0
    cmd = [BLENDER, '-b', '-P', BSCRIPT, '--',
           '--object', os.path.abspath(mesh),
           '--output_dir', os.path.abspath(unit_dir),
           '--resolution', str(resolution),
           '--gpu', '0']
    if transparent:
        cmd.append('--transparent')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)   # 每个 Blender 进程只暴露一张物理卡
    log = os.path.join(logdir, unit + '.log')
    t0 = time.time()
    try:
        with open(log, 'w') as fh:
            r = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout)
        dt = time.time() - t0
        if r.returncode == 0 and is_done(unit_dir):
            return unit, 'ok', dt
        return unit, f'failed(rc={r.returncode})', dt
    except subprocess.TimeoutExpired:
        return unit, 'timeout', time.time() - t0
    except Exception as e:
        return unit, f'error({e})', time.time() - t0


def main():
    """扫描 --root 下的单元 -> 按 GPU×每卡进程数分配槽位 -> 多线程并发渲染 -> 失败重试若干轮。
    成功与否以"返回码 0 且 24 张图齐全"为准；返回码 0=全成功、1=有失败单元。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--gpus', default='5,6,7')
    ap.add_argument('--workers-per-gpu', type=int, default=2)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--transparent', action='store_true')
    ap.add_argument('--only', default=None, help='逗号分隔的单元名，调试用')
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--retry', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true', help='忽略已完成的单元，强制重渲')
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    units = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and not d.startswith('_'))
    if a.only:
        want = set(a.only.split(','))
        units = [u for u in units if u in want]
    logdir = os.path.join(root, '_logs')
    os.makedirs(logdir, exist_ok=True)

    gpus = [g for g in a.gpus.split(',') if g != '']
    slots = [g for g in gpus for _ in range(a.workers_per_gpu)]
    todo = units if a.overwrite else [u for u in units if not is_done(os.path.join(root, u))]
    print(f'单元总数 {len(units)}，待渲染 {len(todo)}，'
          f'GPU {gpus} × {a.workers_per_gpu} = {len(slots)} 进程，'
          f'透明背景={a.transparent} 强制重渲={a.overwrite}', flush=True)

    results = {}
    for attempt in range(a.retry + 1):
        if not todo:
            break
        if attempt:
            print(f'\n--- 重试第 {attempt} 轮，剩余 {len(todo)} ---', flush=True)
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futs = {}
            for i, u in enumerate(todo):
                gpu = slots[i % len(slots)]
                futs[pool.submit(render_one, u, root, gpu, a.resolution,
                                 a.transparent, logdir, a.timeout, a.overwrite)] = u
            done_n = 0
            for f in futs:
                u, st, dt = f.result()
                done_n += 1
                results[u] = st
                print(f'  [{done_n}/{len(todo)}] {u}: {st}'
                      + (f' ({dt/60:.1f} min)' if dt else ''), flush=True)
        todo = [u for u in todo if results.get(u) not in ('ok', 'skipped')]

    ok = sum(1 for v in results.values() if v in ('ok', 'skipped'))
    bad = {u: v for u, v in results.items() if v not in ('ok', 'skipped')}
    print(f'\n完成 {ok}/{len(units)}')
    if bad:
        print(f'失败 {len(bad)}:')
        for u, v in bad.items():
            print(f'  ✗ {u}: {v}')
        with open(os.path.join(root, '_failed.txt'), 'w') as fh:
            for u in bad:
                fh.write(u + '\n')
    else:
        print('全部成功')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())