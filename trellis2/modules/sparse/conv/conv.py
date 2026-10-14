"""稀疏 3D 卷积的统一入口：按 config.CONV 把 init/forward 分派到具体后端实现。

两个门面类只做转发，真正的参数与计算在 conv_{backend}.py 中（约定函数名
sparse_conv3d_init / sparse_conv3d_forward 等，直接改 self）。_backends 缓存已导入模块。
SparseConv3d：普通（子流形）稀疏卷积；SparseInverseConv3d：与前者配对的反卷积/上采样。
"""

from .. import config
import importlib
import torch
import torch.nn as nn
from .. import SparseTensor


_backends = {}  # 后端名 -> 已导入的模块


class SparseConv3d(nn.Module):
    """submanifold 稀疏卷积门面类；stride=1 且 padding=None 时为子流形卷积（输出坐标不变）。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f'..conv_{config.CONV}', __name__)
        _backends[config.CONV].sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_conv3d_forward(self, x)


class SparseInverseConv3d(nn.Module):
    """反卷积门面类：常与 SparseConv3d(stride>1) 配对，用于解码器上采样。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f'..conv_{config.CONV}', __name__)
        _backends[config.CONV].sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_inverse_conv3d_forward(self, x)
