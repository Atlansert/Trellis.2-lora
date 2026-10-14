"""稀疏结构（SS）潜变量数据集与可视化 mixin。

职责：SparseStructureLatent 从 metadata 过滤样本并读取 ss_latent/*.npz 里的 z（按需标准化）；
SparseStructureLatentVisMixin 加载 SS 解码器把潜变量解码成占据体素，渲染 4 视角彩色图。
输入：roots 与 min_aesthetic_score/normalization 等配置；输出：样本 {'x_0': z}。
契约：z 的标准化/反标准化必须使用同一套 normalization 统计；可视化时占用阈值取 z>0。
"""

import os
import json
from typing import *
import numpy as np
import torch
from ..representations import Voxel
from ..renderers import VoxelRenderer
from .components import StandardDatasetBase, ImageConditionedMixin
from .. import models
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


class SparseStructureLatentVisMixin:
    """SS 潜变量可视化 mixin：惰性加载 SS 解码器（本地 ckpt 或预训练）并渲染体素占据。"""

    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        # 优先本地目录 config.json + ckpt，否则从预训练仓库加载 SS 解码器
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        """把 SS 潜变量解码为占据体素：先反归一化，分批解码后拼回；用毕释放解码器省显存。"""
        self._loading_ss_dec()
        ss = []
        if self.normalization:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        # 体素渲染器：512 分辨率 + 4x SSAA 抗锯齿
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        
        # build camera
        # 4 个方位角（每 90°，整体偏 -16°）+ 20° 俯仰；半径 2、fov 30°
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []
        
        # Build each representation
        # 取 z>0 的体素作为占用点；颜色用坐标/resolution 归一化，便于区分空间位置
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords / resolution
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1/resolution,
                coords=coords,
                attrs=color,
                layout={
                    'color': slice(0, 3),
                }
            )
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(rep, ext, intr, colors_overwrite=color)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    稀疏结构潜变量数据集：过滤 metadata 后读取 ss_latent 里的 z（可选标准化）。
    
    Args:
        roots (str): path to the dataset
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        # Locally prepared datasets may not have run the final metadata merge.
        # 中文：本地准备的数据可能没跑最终 metadata 合并——若列不存在就跳过该条件，
        # 只要 ss_latent 目录存在，其文件即为权威来源。
        # When the requested SS latent directory is supplied, its files are
        # authoritative and the base metadata is still sufficient.
        if 'ss_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['ss_latent_encoded'] == True]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        # 读取 ss_latent/<sha256>.npz 中的 z，并按 normalization 做标准化
        latent = np.load(os.path.join(root['ss_latent'], f'{instance}.npz'))
        z = torch.tensor(latent['z']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    单图条件版 SS 数据集：cond 由 components.ImageConditionedMixin 提供。
    """
    pass
    
