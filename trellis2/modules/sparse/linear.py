"""稀疏线性层：直接复用 nn.Linear，只对特征 feats 做投影，坐标不变。"""

import torch
import torch.nn as nn
from . import VarLenTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    """在 VarLenTensor 上作用 nn.Linear：对 feats (N, C) 做线性变换后用 replace 保持坐标。"""
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))
