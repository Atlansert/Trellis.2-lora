"""稀疏归一化层：按 batch 逐元素处理，把该 batch 的所有点当作“空间维”。

做法：用 input.layout[k] 取出第 k 个 batch 的 feats (N_k, C)，重排为 (1, C, N_k) 后
调用父类 GroupNorm/LayerNorm（因此通道维仍是 C），结果写回同样位置。
注意 batch 之间互不混合统计量。*32 版本内部转 fp32 计算再还原 dtype。
"""

import torch
import torch.nn as nn
from ..utils import manual_cast
from . import VarLenTensor
from . import config

__all__ = [
    'SparseGroupNorm',
    'SparseLayerNorm',
    'SparseGroupNorm32',
    'SparseLayerNorm32',
]


class SparseGroupNorm(nn.GroupNorm):
    """逐 batch 的 GroupNorm；点维充当空间维，通道分组内统计。"""
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]  # 第 k 个 batch 的点特征 (N_k, C)
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)  # (1, C, N_k)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)  # 还原为 (N_k, C)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    """逐 batch 的 LayerNorm，输入同样重排为 (1, C, N_k)。"""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    # 先整体转 fp32（manual_cast 作用在 feats 上），算完还原原 dtype
    def forward(self, x: VarLenTensor) -> VarLenTensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)


class SparseLayerNorm32(SparseLayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """
    # 同 SparseGroupNorm32，只是基础层换成 LayerNorm
    def forward(self, x: VarLenTensor) -> VarLenTensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
