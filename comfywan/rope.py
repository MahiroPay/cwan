"""
RoPE (Rotary Position Embedding) implementation for Wan2.2
"""

import torch
from torch import Tensor, nn
from einops import rearrange
import math


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    Compute rotary position embeddings
    
    Args:
        pos: Position tensor
        dim: Dimension
        theta: Theta value
        
    Returns:
        RoPE embeddings
    """
    assert dim % 2 == 0
    device = pos.device
    
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope1(x: Tensor, freqs_cis: Tensor):
    """
    Apply RoPE to input tensor
    
    Args:
        x: Input tensor
        freqs_cis: Frequency tensor
        
    Returns:
        Tensor with RoPE applied
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    """
    Apply RoPE to query and key tensors
    
    Args:
        xq: Query tensor
        xk: Key tensor
        freqs_cis: Frequency tensor
        
    Returns:
        Tuple of (query, key) with RoPE applied
    """
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


class EmbedND(nn.Module):
    """N-dimensional position embedding using RoPE"""
    
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        Args:
            ids: Position IDs with shape [..., n_axes]
            
        Returns:
            Embeddings
        """
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        
        return emb.unsqueeze(1)
