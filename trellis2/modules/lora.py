"""Small, dependency-free LoRA implementation for TRELLIS.2 linear layers.

LoRA 注入：把目标 nn.Linear 冻结（requires_grad=False），并叠加低秩残差
lora_up(lora_down(x)) * (alpha/rank)。适配器权重恒为 fp32 以保证小步更新稳定，
前向时再转回 base 分支的 dtype（AMP/bf16 不受影响）。
apply_lora_to_linear_layers 按“限定名关键字”匹配替换，TRELLIS.2 的目标层为
self_attn.to_qkv / self_attn.to_out / cross_attn.to_q / cross_attn.to_kv / cross_attn.to_out。
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, rank: int = 8,
                 alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank  # 残差缩放系数 alpha/r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始权重（含 bias，若存在）
        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)

        # Keep adapter weights in fp32 for stable small updates.  They are cast
        # to the base output dtype in forward, so AMP/bfloat16 remains intact.
        self.lora_down = nn.Linear(self.in_features, self.rank, bias=False,
                                   device=base_layer.weight.device, dtype=torch.float32)
        self.lora_up = nn.Linear(self.rank, self.out_features, bias=False,
                                 device=base_layer.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)  # up 置零 => 初始等价于原始线性层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pipeline/trainer may keep the frozen base in bfloat16 while adapter
        # matrices stay fp32; align the base branch explicitly for F.linear.
        base = F.linear(x.to(dtype=self.weight.dtype), self.weight, self.bias)
        update = self.lora_up(self.dropout(self.lora_down(x.float())))  # fp32 低秩分支
        return base + update.to(dtype=base.dtype) * self.scale


def _parent(root: nn.Module, name: str):
    # 按点分限定名取出父模块与子属性名，供 setattr 替换
    parts = name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_linear_layers(model: nn.Module, target_keywords: Iterable[str],
                                rank: int = 8, alpha: float = 8.0,
                                dropout: float = 0.0) -> list[str]:
    """Replace matching nn.Linear modules, returning their qualified names."""
    # target_keywords: 限定名需包含的任一关键字（如 "to_qkv"）；返回被替换的限定名列表
    targets = tuple(target_keywords)
    replaced = []
    modules = list(model.named_modules())
    wrapped = tuple(name for name, module in modules if isinstance(module, LoRALinear))
    for name, module in modules:
        # named_modules also visits a wrapped layer's lora_down/lora_up.
        # Those names contain the parent's target keyword: without excluding
        # the whole subtree, a second call recursively injects adapters into
        # adapters and invalidates the checkpoint architecture.
        if any(prefix == '' or name == prefix or name.startswith(prefix + '.') for prefix in wrapped):
            continue
        if isinstance(module, nn.Linear) and any(k in name for k in targets):
            parent, child = _parent(model, name)
            # 就地替换为 LoRA 包装层（原层参数被克隆进冻结分支）
            setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    return replaced
