"""数据集公共组件：标准数据集基类与图像条件 mixin。

职责：StandardDatasetBase 负责解析 roots、加载/合并 metadata.csv（用 sha256 作索引）、
做过滤统计，并暴露 instances 列表；两个图像条件 mixin 负责从渲染视图里取条件图。
输入：roots（JSON 对象或逗号分隔的路径列表）；输出：单个样本 dict（含 'cond' 等键）。
契约：子类必须实现 filter_metadata 与 get_instance；metadata 以 sha256 为索引；
条件图统一裁到 alpha 包围盒并 resize 到 image_size，背景用 alpha 置黑。
"""

from typing import *
import json
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.
    
    标准数据集基类：解析 roots、汇总 metadata.csv、生成 (root, sha256) 实例列表。
    
    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        # roots 支持两种形式：JSON 对象（多来源、每个来源含多个子目录）或逗号分隔的路径列表
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                for _, r in root.items():
                    metadata_path = os.path.join(r, 'metadata.csv')
                    if os.path.exists(metadata_path):
                        part = pd.read_csv(metadata_path).set_index('sha256')
                    elif 'ss_latent' in root:
                        # Locally generated SS latent sets may not yet have
                        # 中文：本地生成的 SS latent 数据可能还没做 metadata 合并，此处用文件名（sha256）
                        # 作为 id 并补上图像条件数据集所需的最小字段
                        # gone through the metadata merge step.  The latent
                        # filenames are SHA256 ids; provide the minimal
                        # fields required by the image-conditioned dataset.
                        ss_root = root['ss_latent']
                        ids = [os.path.splitext(f)[0] for f in os.listdir(ss_root)
                               if f.endswith('.npz')]
                        part = pd.DataFrame({
                            'sha256': ids,
                            'aesthetic_score': 5.5,
                            'cond_rendered': True,
                            'ss_latent_encoded': True,
                        }).set_index('sha256')
                    else:
                        raise FileNotFoundError(metadata_path)
                    metadata = metadata.combine_first(part)
                # 合并各来源的 metadata，并按 sha256 去重（保留第一条），再交给子类做过滤统计
                # Drop duplicate indices that may arise from inconsistent metadata.csv files across subdirectories
                metadata = metadata[~metadata.index.duplicated(keep='first')]
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata.index.values])
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """子类实现：按条件筛掉不合格样本，并返回 (过滤后 metadata, 各阶段计数统计)。"""
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        """子类实现：按 (root, sha256) 读取并组装单个训练样本。"""
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        # 读取失败（坏文件/缺文件）时不中断训练，改为随机换一个样本重试
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(f'Error loading {instance}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    """单图条件 mixin：从 render_cond 的视角帧里随机取一张图作为 cond（乘 alpha 抠背景）。"""

    def __init__(self, roots, *, image_size=518, real_view_prob=0.0, **kwargs):
        self.image_size = image_size
        if not 0.0 <= real_view_prob <= 1.0:
            raise ValueError(f'real_view_prob must be in [0, 1], got {real_view_prob}')
        self.real_view_prob = real_view_prob
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        # 先由上层数据集组装样本，再补上 cond 条件图
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        frames = metadata['frames']
        # 以 real_view_prob 的概率优先挑"真实"视角，否则在所有帧里均匀随机
        real_views = [i for i, frame in enumerate(frames) if frame.get('is_real', False)]
        if real_views and np.random.random() < self.real_view_prob:
            view = np.random.choice(real_views)
        else:
            view = np.random.randint(len(frames))
        metadata = frames[view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        # 用 alpha 通道求包围盒，再以最长边为边长取正方形区域裁剪（居中）
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_hsize = hsize
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        # 乘 alpha 把背景压成 0，得到 [0,1] 的 RGB 条件图 (3,H,W)
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class MultiImageConditionedMixin:
    """多视角条件 mixin：每次随机取 1~max_image_cond_view 个不重复视角，条件为变长视角列表。"""

    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        # 先由上层数据集组装样本，再补上 cond 条件图
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        # 本步随机使用 1..max_image_cond_view 个不重复的视角
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            image = Image.open(image_path)

            # 用 alpha 通道求包围盒，再以最长边为边长取正方形区域裁剪（居中）
            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize
            aug_center = center
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]

            img = image.crop(aug_bbox)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = img.getchannel(3)
            img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            img = img * alpha.unsqueeze(0)

            cond_images.append(img)

        # 注意：cond 是"单元素列表里放一个 (V,3,H,W) 张量"，以匹配上层多视角条件的接口
        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack
