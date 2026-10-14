#!/usr/bin/env python3
"""
扫描 data_gen 下每个子文件夹，用其中的 GLB 渲染 24 个视角图（每 15°），
图片和 transforms.json 生成在 GLB 同目录下。

输入/输出结构（同一目录）:
    data_gen/
        model_001/
            mesh_textured_pbr.glb    ← 输入 GLB
            000_cam_000.png          ← 渲染输出 (yaw=0°)
            001_cam_015.png          ← 渲染输出 (yaw=15°)
            ...
            023_cam_345.png          ← 渲染输出 (yaw=345°)
            transforms.json          ← 相机参数
        model_002/
            ...

用法:
    python scripts/render_views.py --data_dir ./data_gen
"""

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---- 24 视角配置 ----
YAWS_DEG = list(range(0, 360, 15))               # [0, 15, 30, ..., 345]
LABELS = [f'cam_{yaw:03d}' for yaw in YAWS_DEG]  # ['cam_000', 'cam_015', ..., 'cam_345']
NUM_VIEWS = len(YAWS_DEG)                         # 24

# ---- Blender 安装配置 ----
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'


def install_blender():
    """自动下载安装 Blender 3.0.1 (如果尚未安装)。"""
    if os.path.exists(BLENDER_PATH):
        return

    print('=' * 60)
    print('Blender 3.0.1 未安装，正在自动安装...')
    print('=' * 60)

    deps = 'libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1'
    os.system(f'sudo apt-get update -qq 2>/dev/null')
    os.system(f'sudo apt-get install -y -qq {deps} 2>/dev/null')

    print(f'下载 Blender (约 200MB)...')
    ret = os.system(f'wget -q --show-progress {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
    if ret != 0:
        print(f'[错误] Blender 下载失败，请检查网络或手动安装')
        sys.exit(1)

    print(f'解压...')
    os.system(f'tar -xf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

    if os.path.exists(BLENDER_PATH):
        print('Blender 安装成功!\n')
    else:
        print(f'[错误] Blender 安装失败，请手动安装到 {BLENDER_PATH}')
        sys.exit(1)


def find_glb_in_subdirs(data_dir):
    """
    扫描 data_dir 下每个子文件夹，找 GLB/OBJ 文件。
    返回: [(subdir_name, glb_path), ...]
    """
    entries = []
    data_path = Path(data_dir)
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue
        glbs = (list(subdir.glob('*.glb')) + list(subdir.glob('*.GLB')) +
                list(subdir.glob('*.obj')) + list(subdir.glob('*.OBJ')))
        if glbs:
            entries.append((subdir.name, str(glbs[0])))
        else:
            print(f'[警告] {subdir.name}: 无 GLB/OBJ 文件，跳过')
    return entries


def render_one(name, glb_path, blender_script):
    """用 GLB 渲染 24 个视角图，输出到 GLB 同目录。"""
    out_dir = os.path.dirname(glb_path)

    # 检查是否已完成
    done_marker = os.path.join(out_dir, 'transforms.json')
    if os.path.exists(done_marker):
        all_imgs = all(
            os.path.exists(os.path.join(out_dir, f'{i:03d}_{label}.png'))
            for i, label in enumerate(LABELS)
        )
        if all_imgs:
            return {'name': name, 'status': 'skipped'}

    cmd = [
        BLENDER_PATH, '-b', '-P', blender_script,
        '--',
        '--object', os.path.abspath(glb_path),
        '--output_dir', os.path.abspath(out_dir),
        '--resolution', '1024',
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode == 0 and os.path.exists(done_marker):
        return {'name': name, 'status': 'ok'}
    else:
        err = r.stderr.strip().split('\n')[-1] if r.stderr.strip() else f'returncode={r.returncode}'
        return {'name': name, 'status': 'failed', 'error': err}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=f'用 data_gen 中的 GLB 渲染 {NUM_VIEWS} 个视角图 (每 15°, 纯白背景)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录，内含子文件夹 (每个子文件夹有 GLB)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行 Blender 进程数 (默认: 4)')
    parser.add_argument('--only', type=str, default=None,
                        help='只处理指定子文件夹名 (逗号分隔，调试用)')
    args = parser.parse_args()

    # 定位 Blender 脚本
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blender_script = os.path.join(
        repo_root, 'data_toolkit', 'blender_script', 'render_view.py')
    if not os.path.exists(blender_script):
        print(f'[错误] Blender 脚本不存在: {blender_script}')
        sys.exit(1)

    # Step 1: 安装 Blender
    install_blender()

    # Step 2: 扫描子文件夹
    entries = find_glb_in_subdirs(args.data_dir)
    if not entries:
        print(f'[错误] {args.data_dir} 子文件夹中没有 GLB/OBJ 文件')
        sys.exit(1)

    if args.only:
        only_names = set(args.only.split(','))
        entries = [(n, p) for n, p in entries if n in only_names]
        print(f'[调试模式] 仅处理: {args.only}')
        if not entries:
            print(f'[错误] 指定子文件夹不在目录中')
            sys.exit(1)

    print(f'找到 {len(entries)} 个子文件夹')
    print(f'数据目录: {args.data_dir}')
    print(f'每个模型渲染 {NUM_VIEWS} 个视角 (0°~345°, 每 15°)，输出到 GLB 同目录')
    print(f'分辨率: 1024×1024, 纯白背景\n')

    # Step 3: 并行渲染
    results = {'ok': 0, 'skipped': 0, 'failed': 0}
    failed_names = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for name, glb_path in entries:
            f = pool.submit(render_one, name, glb_path, blender_script)
            futures[f] = name

        with tqdm(total=len(entries), desc='渲染进度') as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r['status'] == 'ok':
                    results['ok'] += 1
                elif r['status'] == 'skipped':
                    results['skipped'] += 1
                else:
                    results['failed'] += 1
                    failed_names.append({'name': r['name'], 'error': r.get('error', 'unknown')})
                pbar.set_postfix(
                    ok=results['ok'], skip=results['skipped'], fail=results['failed'])
                pbar.update(1)

    print(f'\n{"=" * 60}')
    print(f'渲染完成: {results["ok"]} 成功, {results["skipped"]} 跳过, {results["failed"]} 失败')
    if failed_names:
        print(f'失败模型:')
        for fn in failed_names:
            print(f'  ✗ {fn["name"]} — {fn["error"]}')
        only_names = ','.join(fn['name'] for fn in failed_names)
        print(f'  重试: python scripts/render_views.py --data_dir {args.data_dir} '
              f'--only "{only_names}"')

    print(f'\n下一步:')
    print(f'  1. 查看 {args.data_dir}/<子文件夹>/ 中的 {NUM_VIEWS} 张渲染图')
    print(f'  2. 选出每个模型「车头朝右」的视角')
    print(f'  3. 运行 organize_views.py 导入训练数据')


if __name__ == '__main__':
    main()
