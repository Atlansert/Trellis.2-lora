"""Flexi Dual-Grid VAE：在稀疏 U-Net VAE 基础上改造成“双重网格(flexible dual grid)”表示。

- 编码输入 6 通道：3 通道顶点相对坐标 + 3 通道相交(intersected)标志，均中心化到 0。
- 解码输出 7 通道：0:3 顶点坐标（sigmoid 到 [-margin, 1+margin]）、3:6 相交 logits、
  6:7 每个 voxel 的 quad_lerp（softplus 后 >=0），分辨率由 resolution 决定（512/1024）。
- forward 训练时返回 (mesh, vertices, intersected_logits, subs_gt, subs)，
  推理时把 mesh 作为第 0 项返回；顶点均以 [-0.5,0.5]³ 单位立方体为参考系。
"""
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    """双网格 VAE 编码器：输入 6 通道（顶点偏移 + 相交标志），输出 SLat 潜变量。"""
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__(
            6,  # 输入通道 = 3 顶点相对坐标 + 3 相交标志
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        # 顶点坐标与相交标志都减 0.5，使取值中心化到 0 附近后再拼接送入编码器
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    """双网格 VAE 解码器：从 SLat 解码 7 通道场，再转成 Mesh（须先设好 resolution）。"""
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        
        super().__init__(
            7,  # 输出通道 = 3 顶点偏移 + 3 相交 logits + 1 quad_lerp
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        # 训练：用 sigmoid 把顶点回归到 [-voxel_margin, 1+voxel_margin]，相交用 gt 监督
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            # 推理：相交直接用 logits > 0 二值化，无需 gt；aabb 固定为单位立方体
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected = h.replace(h.feats[..., 3:6] > 0)
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=False
            )) for v, i, q in zip(vertices, intersected, quad_lerp)]
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
