"""torchsparse 后端的稀疏卷积（stride=1 时保持 layout，否则 layout 需重算）。"""

import torch
import torch.nn as nn
from .. import SparseTensor
import torchsparse


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    # torchsparse 不用 indice_key，padding 固定为 0
    self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias)


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    out = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
    out._spatial_cache = x._spatial_cache
    out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])  # 下采样后尺度放大
    return out


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    # transposed=True 即反卷积（上采样）
    self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias, transposed=True)


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    out = self.conv(x.data)        
    new_shape = [x.shape[0], self.conv.out_channels]
    out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
    out._spatial_cache = x._spatial_cache
    out._scale = tuple([s / stride for s, stride in zip(x._scale, self.conv.stride)])  # 上采样后尺度缩小
    return out
