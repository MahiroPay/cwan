"""
Lightweight operations module - replacement for comfy.ops
Provides standard PyTorch layers without weight initialization disabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ramtorch import Linear as RTLinear


class Conv3d(nn.Conv3d):
    """Standard Conv3d"""
    pass


class Conv2d(nn.Conv2d):
    """Standard Conv2d"""
    pass


class Linear(RTLinear):
# class Linear(nn.Linear):
    """Standard Linear"""
    pass


class LayerNorm(nn.LayerNorm):
    """Standard LayerNorm"""
    pass


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        if self.weight is not None:
            x = x * self.weight
        return x


class Embedding(nn.Embedding):
    """Standard Embedding"""
    pass


# Create a module that mimics comfy.ops.disable_weight_init
class DisableWeightInit:
    """Context manager that returns standard operations"""
    Conv3d = Conv3d
    Conv2d = Conv2d
    Linear = Linear
    LayerNorm = LayerNorm
    RMSNorm = RMSNorm
    Embedding = Embedding


disable_weight_init = DisableWeightInit()
