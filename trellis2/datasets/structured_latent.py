"""结构化潜变量（SLat）数据集基类与可视化 mixin。

职责：SLat 从 npz 读取稀疏潜变量（coords + feats），按 metadata 的 aesthetic/token 数过滤，
并把样本拼成稀疏张量批次（collate_fn：coords 首列插入 batch 内索引、layout 记录每样本切片）；
SLatVisMixin 负责加载 SLat 解码器做多视角渲染可视化。
输入：roots 与 latent_key/normalization 等配置；输出：{'coords','feats'} 样本或含 x_0 的批次。
契约：feats 的标准化必须用同一套 normalization 统计（解码前要反归一化）；collate_fn 返回
list（split_size 给定，用于梯度累积）或单个 pack；坐标轴约定 raw 轴 = (depth, length, up)。
"""

import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.data_utils import load_balanced_group_indices


class SLatVisMixin:
    """SLat 可视化 mixin：惰性加载 SLat 解码器（本地 ckpt 或预训练），把潜变量渲染成多视角图。"""

    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        
    def _loading_slat_dec(self):
        # 优先按本地目录的 config.json + ckpt 构建解码器，否则从预训练仓库加载
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.slat_dec
        self.slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        """把潜变量解码为网格表示：先反归一化，再分批解码；用完整批后释放解码器以省显存。"""
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            reps.append(self.slat_dec(z[i:i+batch_size]))
        reps = sum(reps, [])
        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
        # Build camera
        # 4 个方位角（间隔 90°，整体随机偏移）+ 随机俯仰；相机距离 2、fov 40°
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        # 渲染彩色图并拼成 2x2（每格 512x512）
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        images = torch.stack(images)
            
        return images
    
    
class SLat(SLatVisMixin, StandardDatasetBase):
    """
    structured latent V2 dataset
    
    结构化潜变量数据集：按 latent_key 读取稀疏潜变量，过滤后拼成稀疏张量批次。
    
    Args:
        roots (str): path to the dataset
        min_aesthetic_score (float): minimum aesthetic score
        max_tokens (int): maximum number of tokens
        latent_key (str): key of the latent to be used
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        latent_key: str = 'shape_latent',
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.latent_key = latent_key
        self.value_range = (0, 1)   # 供可视化时的取值范围
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        # 预取每个实例的 token 数，供采样/统计使用
        self.loads = [self.metadata.loc[sha256, f'{latent_key}_tokens'] for _, sha256 in self.instances]
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        # 三级过滤：已编码出潜变量 -> 美学分达标 -> token 数不超过上限（逐级记录剩余数量）
        stats = {}
        metadata = metadata[metadata[f'{self.latent_key}_encoded'] == True]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata[f'{self.latent_key}_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        # 读取该实例的稀疏潜变量 npz；feats 按配置的 mean/std 标准化
        data = np.load(os.path.join(root[self.latent_key], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        """把多个样本拼成 SparseTensor 批次：coords 首列写 batch 内样本索引，layout 记录各样本切片。"""
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                # 首列插入样本索引（稀疏张量的约定：coords = [batch_idx, x, y, z]）
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # collate other data
            # 其余键按类型聚合：Tensor 直接 stack，list 展平拼接，其它则收集成列表
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs


class ImageConditionedSLat(ImageConditionedMixin, SLat):
    """
    Image conditioned structured latent dataset
    单图条件版：cond 由 components.ImageConditionedMixin 从渲染视角里取。
    """
    pass
