"""用 Blender 批量渲染条件视角（多视角 RGB/A 图 + transforms.json）。

- 用途：数据重建流水线的渲染步骤：为每个已下载资产在球面上采样若干相机位姿并渲染，
  产物是供 SS/shape/PBR latent 与多视角视角对齐使用的条件渲染图。
- 输入：--root（含 metadata.csv）、--download_root/raw/ 的本地资产、--num_cond_views 视角数（默认 16）。
- 输出：<render_cond_root>/renders_cond/<sha256>/（各视角图 + transforms.json，含 yaw_deg 等帧信息）
  以及 renders_cond/new_records/part_<rank>.csv（增量进度）。
- 关键约定：相机方向用 Hammersley 球面序列采样（utils.sphere_hammersley_sequence），
  半径/视场在 fov 10°~70° 之间随机采样；渲染引擎 CYCLES、条件图分辨率 1024；
  以 transforms.json 是否存在判断对象已完成（可中断续跑）。
- 用法：python render_cond.py <dataset_module> --root <dir> [--num_cond_views 16] [--max_workers 8]
"""
import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from utils import sphere_hammersley_sequence


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

# 若 /tmp 下没有 Blender 3.0.1（含运行库依赖）就下载安装；已有则直接复用。
def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


# 单对象任务：采样条件相机（Hammersley 球面方向 + 随机半径/视场），调用
# blender_script/render_cond.py 渲染；以 transforms.json 是否生成作为成功判据。
def _render_cond(file_path, metadatum, root, num_cond_views):
    sha256 = metadatum['sha256']
    # Build conditional view camera
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_cond_views):
        y, p = sphere_hammersley_sequence(i, num_cond_views, offset)
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 10, 70
    # 由视场角反推相机半径（保持物体外接球刚好入画），再做面积守恒的均匀采样：
    # 先采样 1/k^2 得半径，再由半径反解 fov。
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    # 注意：radius/fov 是 1e6 个候选，下面的 zip 只取前 num_cond_views 个与 yaw/pitch 配对。
    cond_views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_cond.py'),
        '--',
        '--object', os.path.expanduser(file_path),
        '--cond_views', json.dumps(cond_views),
        '--cond_resolution', '1024',
        '--cond_output_folder', os.path.join(root, 'renders_cond', sha256),
        '--engine', 'CYCLES',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    if os.path.exists(os.path.join(root, 'renders_cond', sha256, 'transforms.json')):
        return {'sha256': sha256, 'cond_rendered': True}


# CLI 入口：以 datasets.<sys.argv[1]> 为数据集适配模块；import utils 依赖当前工作目录（脚本同目录）。
if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--render_cond_root', type=str, default=None,
                        help='Directory to save the mesh dumps')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_cond_views', type=int, default=16,
                        help='Number of conditional views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.render_cond_root = opt.render_cond_root or opt.root

    os.makedirs(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.render_cond_root, 'renders_cond', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.render_cond_root, 'renders_cond', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[(metadata['cond_rendered'] != True)]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
        tqdm(total=len(metadata), desc="Filtering existing objects") as pbar:
        def check_sha256(sha256):
            if os.path.exists(os.path.join(opt.render_cond_root, 'renders_cond', sha256, 'transforms.json')):
                records.append({'sha256': sha256, 'cond_rendered': True})
            pbar.update()
        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)
    existing_sha256 = set(r['sha256'] for r in records)
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]

    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render_cond, root=opt.render_cond_root, num_cond_views=opt.num_cond_views)
    cond_rendered = dataset_utils.foreach_instance(metadata, opt.render_cond_root, func, max_workers=opt.max_workers, desc='Rendering objects')
    cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    # 每个 rank 写自己的进度分片。
    cond_rendered.to_csv(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records', f'part_{opt.rank}.csv'), index=False)
