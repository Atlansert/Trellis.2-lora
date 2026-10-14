"""RoPE（旋转位置编码）实现（dense 版，用于图像 token 序列）。

思路：把整数坐标 indices(P 维) 通过一组频率映射成复数相位 phases，
再把 q/k 的最后 head_dim 维两两视作复数相乘，实现按位置旋转。
freqs 长度 = head_dim/2/dim（dim 为坐标维数，默认 3 = 体素 xyz）；
head_dim/2 以外的高频维相位为 1+0j（即不旋转，等价补零）。
"""

from typing import *
import torch
import torch.nn as nn


class RotaryPositionEmbedder(nn.Module):
    """由坐标 indices 生成 RoPE 复数相位，并提供对 q/k 施加旋转的静态方法。"""
    def __init__(
        self, 
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim  # 每个坐标维占用的频率个数
        # 频率从 rope_freq[0] 递减到 rope_freq[0]/rope_freq[1]（几何级数）
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [M]（单坐标维），返回 [M, freq_dim] 的单位复数 e^{i*idx*freq}
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
    
    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        # x: [..., N, H, C]，C 为偶数；两两配对成复数后乘相位再还原
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        # 展平所有坐标算相位，再还原成 [..., N, dim*freq_dim]（即 head_dim/2 维）
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
                # 不足 head_dim/2 的部分用相位 0 补齐（该维度不旋转）
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device)
                )], dim=-1)
        return phases