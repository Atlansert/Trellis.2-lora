"""形状结构化潜变量（SLat）数据集：加载 shape_latent 并支持可视化。

职责：在 SLat 基类之上固定 latent_key='shape_latent'，注入形状 SLat 解码器用于
把潜变量解码成网格并渲染预览；另提供图像条件/多视角条件版本。
输入：数据集根目录 roots 与分辨率、token 上限、归一化统计等；输出：训练样本（潜变量 + 条件）。
契约：潜变量的归一化/裁剪依赖 SLat 基类；可视化需先加载解码器（本地 ckpt 或预训练）。
"""

import os
import json
from typing import *
import numpy as np
import torch
from .. import models
from .components import ImageConditionedMixin, MultiImageConditionedMixin
from ..modules.sparse import SparseTensor
from .structured_latent import SLatVisMixin, SLat
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics


class SLatShapeVisMixin(SLatVisMixin):
    """形状 SLat 可视化 mixin：加载形状解码器，把潜变量解码成网格并渲染多视角法线图。"""

    def _loading_slat_dec(self):
        # 惰性加载形状 SLat 解码器：优先本地 ckpt 路径，否则从预训练仓库取；随后设分辨率并转 GPU/eval
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        decoder.set_resolution(self.resolution)
        self.slat_dec = decoder.cuda().eval()

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        """把潜变量解码为网格表示，再用 4 个环绕视角渲染法线图拼成 2x2 预览（返回 [N,3,H,W]）。"""
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
        # build camera
        # 4 个方位角（每 90°，整体偏 -16°）+ 20° 俯仰；半径 2、fov 30°
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)
        
        # render
        # 每个样本渲染法线图并拼成 2x2 画布（每格 512x512）
        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['normal']
            images.append(image)
        images = torch.stack(images)
        return images
    
    
class SLatShape(SLatShapeVisMixin, SLat):
    """
    structured latent for shape generation
    
    形状 SLat 数据集：固定使用 shape_latent 作为潜变量键，并挂上形状解码器做可视化。
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the shape
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
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        super().__init__(
            roots,
            min_aesthetic_score=min_aesthetic_score,
            max_tokens=max_tokens,
            latent_key='shape_latent',
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        self.resolution = resolution


class ImageConditionedSLatShape(ImageConditionedMixin, SLatShape):
    """
    Image conditioned structured latent for shape generation
    单图条件版：图像条件的编码由 components.ImageConditionedMixin 提供。
    """
    pass


class MultiImageConditionedSLatShape(MultiImageConditionedMixin, SLatShape):
    """
    Multi-image conditioned structured latent for shape generation.

    Randomly samples 1~max_image_cond_view views from renders_cond/{instance}/
    per training step. The model learns to map multiple viewpoints to the same
    3D shape latent, improving generalization across viewing angles.

    多视角条件版：每步从渲染条件目录里随机取 1~max_image_cond_view 个视角，
    让模型学会用多个视角映射到同一个 3D 潜变量。

    Args:
        max_image_cond_view (int): Maximum number of views to sample per step.
            Default 4. Set higher (e.g. 6-8) for datasets with many views.
    """
    pass
