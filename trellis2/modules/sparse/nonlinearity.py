"""稀疏激活层：沿 feats 做逐点非线性，稀疏结构（coords/layout）原样保留。"""

import torch
import torch.nn as nn
from . import VarLenTensor

__all__ = [
    'SparseReLU',
    'SparseSiLU',
    'SparseGELU',
    'SparseActivation'
]


class SparseReLU(nn.ReLU):
    """逐点 ReLU，仅作用于 feats。"""
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))
    

class SparseSiLU(nn.SiLU):
    """逐点 SiLU，仅作用于 feats。"""
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    """逐点 GELU，仅作用于 feats。"""
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseActivation(nn.Module):
    """把任意 nn.Module 激活函数套用到 feats 上。"""
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(self.activation(input.feats))
    
