"""
Wan2.2 Model Architecture
Original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

from .rope import EmbedND, apply_rope1
from .utils import optimized_attention, pad_to_patch_size, cast_to
from .ops import disable_weight_init as ops


def sinusoidal_embedding_1d(dim, position):
    """Create sinusoidal timestep embeddings"""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def repeat_e(e, x):
    """Repeat embedding e to match x's batch size"""
    if e.shape[0] != x.shape[0]:
        e = e.repeat(x.shape[0], 1, 1)
    return e


class WanSelfAttention(nn.Module):
    """Self-attention module with RoPE"""
    
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, kv_dim=None, operations=ops):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        if kv_dim is None:
            kv_dim = dim

        self.q = operations.Linear(dim, dim)
        self.k = operations.Linear(kv_dim, dim)
        self.v = operations.Linear(kv_dim, dim)
        self.o = operations.Linear(dim, dim)
        self.norm_q = operations.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = operations.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs, transformer_options={}):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn_q(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            return apply_rope1(q, freqs)

        def qkv_fn_k(x):
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            return apply_rope1(k, freqs)

        q = qkv_fn_q(x)
        k = qkv_fn_k(x)

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            self.v(x).view(b, s, n * d),
            heads=self.num_heads,
            transformer_options=transformer_options,
        )

        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross attention"""
    
    def forward(self, x, context, transformer_options={}, **kwargs):
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, -1, n * d),
            v.view(b, -1, n * d),
            heads=self.num_heads,
            transformer_options=transformer_options,
        )
        
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    """Wan attention block with self-attention, cross-attention, and FFN"""
    
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        operations=ops,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Attention layers
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, operations=operations)
        
        if cross_attn_type == 't2v_cross_attn':
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), cross_attn_norm, eps, operations=operations)
        else:
            raise NotImplementedError(f"Cross attention type {cross_attn_type} not implemented")
        
        # Normalization layers
        self.norm1 = operations.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = operations.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = operations.LayerNorm(dim, eps, elementwise_affine=True) 
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            operations.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            operations.Linear(ffn_dim, dim)
        )

        # Modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(self, x, e, freqs, context, context_img_len=257, transformer_options={}):
        # Modulation
        if e.ndim < 4:
            e = (cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

        # Self-attention
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            freqs,
            transformer_options=transformer_options
        )
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # Cross-attention & FFN
        x = x + self.cross_attn(self.norm3(x), context, transformer_options=transformer_options)
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class Head(nn.Module):
    """Output head"""
    
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operations=ops):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = operations.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = operations.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        if e.ndim < 3:
            e = (cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e.unsqueeze(2)).unbind(2)

        x = self.head(torch.addcmul(repeat_e(e[0], x), self.norm(x), 1 + repeat_e(e[1], x)))
        return x


class WanModel(torch.nn.Module):
    """
    Wan2.2 diffusion backbone
    
    Args:
        model_type: Model variant ('t2v' for text-to-video)
        patch_size: 3D patch dimensions (t_patch, h_patch, w_patch)
        text_len: Fixed length for text embeddings
        in_dim: Input video channels (C_in)
        dim: Hidden dimension of the transformer
        ffn_dim: Intermediate dimension in feed-forward network
        freq_dim: Dimension for sinusoidal time embeddings
        text_dim: Input dimension for text embeddings
        out_dim: Output video channels (C_out)
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        window_size: Window size for local attention
        qk_norm: Enable query/key normalization
        cross_attn_norm: Enable cross-attention normalization
        eps: Epsilon value for normalization layers
    """
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=48,  # Wan2.2 uses 48 channels
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        device=None,
        dtype=None,
        operations=ops,
    ):
        super().__init__()
        self.dtype = dtype
        
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim),
            nn.SiLU(),
            operations.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, dim * 6)
        )

        # Transformer blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                operations=operations
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.head = Head(dim, out_dim, patch_size, eps, operations=operations)

        # RoPE embedder
        d = dim // num_heads
        self.rope_embedder = EmbedND(
            dim=d,
            theta=10000.0,
            axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        )

    def rope_encode(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None, device=None, dtype=None):
        """Encode positional information using RoPE"""
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(
            t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(
            0, h_len - 1, steps=steps_h, device=device, dtype=dtype
        ).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(
            0, w_len - 1, steps=steps_w, device=device, dtype=dtype
        ).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return freqs

    def forward_orig(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        """
        Forward pass through the diffusion model
        
        Args:
            x: Input video tensors [B, C_in, F, H, W]
            t: Diffusion timesteps [B]
            context: Text embeddings [B, L, C]
            freqs: Rope freqs
            
        Returns:
            Denoised video tensors [B, C_out, F, H, W]
        """
        # Patch embedding
        x = self.patch_embedding(x.to(self.patch_embedding.weight.dtype)).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x.dtype)
        )
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # Text embeddings
        context = self.text_embedding(context)
        

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(
                x, e=e0, freqs=freqs, context=context,
                transformer_options=transformer_options
            )

        # Output head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        """
        Main forward pass with padding and RoPE encoding
        
        Args:
            x: Input latent [B, C, T, H, W]
            timestep: Timestep [B]
            context: Text embeddings [B, L, C]
            
        Returns:
            Output latent [B, C, T, H, W]
        """
        bs, c, t, h, w = x.shape
        x = pad_to_patch_size(x, self.patch_size)

        t_len = t
        if time_dim_concat is not None:
            time_dim_concat = pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = x.shape[2]

        freqs = self.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype)
        return self.forward_orig(
            x, timestep, context,
            clip_fea=clip_fea,
            freqs=freqs,
            transformer_options=transformer_options,
            **kwargs
        )[:, :, :t, :h, :w]

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings
        
        Args:
            x: Patchified features [B, L, C_out * prod(patch_size)]
            grid_sizes: Original spatial-temporal grid dimensions
            
        Returns:
            Reconstructed video tensors [B, C_out, F, H, W]
        """
        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u
