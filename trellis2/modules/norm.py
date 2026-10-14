"""归一化层工具：统一以 float32 计算、再转回原 dtype 的 LayerNorm/GroupNorm。

适用场景：混合精度（bf16/fp16）训练或推理时，归一化的统计量对精度敏感，
故 forward 内部先 manual_cast 到 fp32，算完再转回输入 dtype。
ChannelLayerNorm32 把通道维移到最后一维做 LayerNorm（即对通道归一化）。
"""

import torch
import torch.nn as nn
from .utils import manual_cast


class LayerNorm32(nn.LayerNorm):
    """LayerNorm 的 fp32 包装：fp32 计算后还原输入 dtype。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    """对通道维做 LayerNorm：把通道维搬到最后再调用父类，最后搬回原位。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()  # (B, C, ...) -> (B, ..., C)
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()  # (B, ..., C) -> (B, C, ...)
        return x
    