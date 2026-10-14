"""Small, dependency-free LoRA implementation for TRELLIS.2 linear layers."""

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
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pipeline/trainer may keep the frozen base in bfloat16 while adapter
        # matrices stay fp32; align the base branch explicitly for F.linear.
        base = F.linear(x.to(dtype=self.weight.dtype), self.weight, self.bias)
        update = self.lora_up(self.dropout(self.lora_down(x.float())))
        return base + update.to(dtype=base.dtype) * self.scale


def _parent(root: nn.Module, name: str):
    parts = name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_linear_layers(model: nn.Module, target_keywords: Iterable[str],
                                rank: int = 8, alpha: float = 8.0,
                                dropout: float = 0.0) -> list[str]:
    """Replace matching nn.Linear modules, returning their qualified names."""
    targets = tuple(target_keywords)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and any(k in name for k in targets):
            parent, child = _parent(model, name)
            setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    return replaced
