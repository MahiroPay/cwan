"""
Utility functions for Wan2.2 model
"""

import torch
from einops import rearrange


def rms_norm(x, weight, eps):
    """Root Mean Square normalization"""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    x = x / rms
    return x * weight


def pad_to_patch_size(x, patch_size):
    """Pad tensor to be divisible by patch size"""
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    
    _, _, t, h, w = x.shape
    t_pad = (patch_size[0] - t % patch_size[0]) % patch_size[0]
    h_pad = (patch_size[1] - h % patch_size[1]) % patch_size[1]
    w_pad = (patch_size[2] - w % patch_size[2]) % patch_size[2]
    
    if t_pad > 0 or h_pad > 0 or w_pad > 0:
        x = torch.nn.functional.pad(x, (0, w_pad, 0, h_pad, 0, t_pad))
    
    return x


def cast_to(tensor, dtype=None, device=None):
    """Cast tensor to dtype and device"""
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def optimized_attention(q, k, v, heads, skip_reshape=False, mask=None, transformer_options=None):
    """
    Simplified attention computation using PyTorch's scaled_dot_product_attention
    """
    if not skip_reshape:
        b, s_q, d = q.shape
        s_k = k.shape[1]
        s_v = v.shape[1]
        head_dim = d // heads
        q = q.view(b, s_q, heads, head_dim).transpose(1, 2)
        k = k.view(b, s_k, heads, head_dim).transpose(1, 2)
        v = v.view(b, s_v, heads, head_dim).transpose(1, 2)
    else:
        b, heads, s_q, head_dim = q.shape
    
    # Use PyTorch's scaled_dot_product_attention
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False
    )
    
    if not skip_reshape:
        out = out.transpose(1, 2).contiguous().view(b, s_q, -1)
    else:
        out = out.transpose(1, 2).contiguous().view(b, s_q, heads * head_dim)
    
    return out
