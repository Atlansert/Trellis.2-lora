"""下载原始 3D 资产（GLB/OBJ 等），是数据流水线的第一步。

- 用途：按 metadata.csv 逐对象下载原始资产文件，并回填 local_path 等标记列。
- 输入：--root（含 metadata.csv）、数据集适配模块（命令行第 1 个位置参数，
  形如 datasets.<name>，提供 add_args/download）、可选 --instances 指定对象、--check_only 只校验不下载。
- 输出：<download_root>/raw/ 下的资产文件 + raw/new_records/part_<rank>.csv（增量进度）。
- 关键约定：默认只处理 local_path 为空（尚未下载）的对象；支持 --rank/--world_size 分片与美学分过滤。
- 用法：python download.py <dataset_module> --root <dir> [--download_root <dir>] [--check_only]
"""
import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict

# CLI 入口：以 datasets.<sys.argv[1]> 为数据集适配模块，实际下载交给 dataset_utils.download。
if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to download the objects')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check if the objects are already downloaded')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root

    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(opt.download_root, exist_ok=True)
    os.makedirs(os.path.join(opt.download_root, 'raw', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'local_path' in metadata.columns:
            # 只挑还没下载过的对象（local_path 为空）。
            metadata = metadata[metadata['local_path'].isna()]
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
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    downloaded = dataset_utils.download(metadata, **opt)
    # 每个 rank 写自己的进度分片。
    downloaded.to_csv(os.path.join(opt.download_root, 'raw', 'new_records', f'part_{opt.rank}.csv'), index=False)
