"""
Text encoder for Wan2.2 using T5
ComfyUI UMT5-XXL implementation (standalone version)
"""

import torch
import torch.nn as nn
import math
import os
import json


# ============================================================================
# T5 Layer Norm
# ============================================================================
class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(x.dtype) * x


# ============================================================================
# T5 Feed-Forward Layers
# ============================================================================
class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = nn.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


# ============================================================================
# T5 Attention
# ============================================================================
class T5Attention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.q = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.inner_dim = inner_dim

        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device, dtype=dtype)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)
        
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, query_length, key_length, device):
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, x, mask=None, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = q.shape
        head_dim = self.inner_dim // self.num_heads
        
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Add relative position bias
        if self.relative_attention_bias is not None:
            if past_bias is None:
                past_bias = self.compute_bias(seq_len, seq_len, x.device)
            scores = scores + past_bias.to(scores.dtype)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax and attend
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)
        attn_output = self.o(attn_output)
        
        return attn_output, past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, mask=None, past_bias=None):
        normed_hidden_states = self.layer_norm(x)
        attention_output, past_bias = self.SelfAttention(normed_hidden_states, mask=mask, past_bias=past_bias)
        x = x + attention_output
        return x, past_bias


# ============================================================================
# T5 Block and Stack
# ============================================================================
class T5Block(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, mask=None, past_bias=None):
        x, past_bias = self.layer[0](x, mask, past_bias)
        x = self.layer[1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads, dtype, device, offload_device):
        super().__init__()
        self.offload_device = offload_device
        self.device = device
        self.block = nn.ModuleList(
            [T5Block(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias=(i == 0), dtype=dtype, device=offload_device)
             for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, attention_mask=None):
        mask = None
        x.to(self.device)
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        past_bias = None
        for layer in self.block:
            # move layer to main device
            layer = layer.to(self.device)
            x, past_bias = layer(x, mask, past_bias)
            # move layer back to offload device
            layer = layer.to(self.offload_device)
        
        x = self.final_layer_norm(x)
        return x


# ============================================================================
# T5 Model
# ============================================================================
class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, offload_device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        model_dim = config_dict["d_model"]
        inner_dim = config_dict["d_kv"] * config_dict["num_heads"]

        self.encoder = T5Stack(
            self.num_layers, 
            model_dim, 
            inner_dim, 
            config_dict["d_ff"], 
            config_dict["num_heads"], 
            dtype, 
            device,
            offload_device
        )
        self.dtype = dtype
        self.shared = nn.Embedding(config_dict["vocab_size"], model_dim, device=device, dtype=dtype)

    def get_input_embeddings(self):
        return self.shared

    def forward(self, input_ids, attention_mask):
        x = self.shared(input_ids)
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x)  # Fix for fp8 T5 base
        return self.encoder(x, attention_mask=attention_mask)


# ============================================================================
# Wan UMT5-XXL Text Encoder (ComfyUI-Compatible Implementation)
# ============================================================================
class WanT5TextEncoder:
    """
    Wan2.2 text encoder using UMT5-XXL
    
    This implementation matches ComfyUI's text encoding architecture:
    - Uses encode_token_weights() for weighted token embeddings
    - Returns (cond, pooled, extra) tuple format
    - Supports attention masks in extra dict
    - Compatible with encode_from_tokens_scheduled workflow
    """
    
    def __init__(self, device="cuda", dtype=torch.float16, tokenizer_path=None, offload_device="cpu"):
        """
        Initialize UMT5-XXL text encoder
        
        Args:
            device: Device to run model on
            dtype: Data type for model
            tokenizer_path: Path to SentencePiece model (optional, will use transformers fallback)
            offload_device: Device to offload layers to when not in use
        """
        self.device = device
        self.offload_device = offload_device
        self.dtype = dtype
        
        # UMT5-XXL configuration
        config = {
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "num_heads": 64,
            "num_layers": 24,
            "vocab_size": 256384,
            "model_type": "umt5"
        }
        
        # Initialize tokenizer
        print(f"Initializing UMT5-XXL tokenizer...")
        self.tokenizer = None
        self.tokenizer_path = str(tokenizer_path) if tokenizer_path else None
        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            try:
                import sentencepiece
                self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizer_path)
                print(f"Loaded SentencePiece tokenizer from {self.tokenizer_path}")
                self.use_transformers = False
            except ImportError:
                print("sentencepiece not installed, falling back to transformers")
                self.tokenizer = None
        
        # Fallback to transformers tokenizer
        if self.tokenizer is None:
            from transformers import T5Tokenizer
            print("Loading transformers T5Tokenizer (fallback)...")
            self.tokenizer = T5Tokenizer.from_pretrained("google/umt5-xxl")
            self.use_transformers = True
        
        print(f"Initializing UMT5-XXL model...")
        self.model = T5(config, dtype=dtype, device=device, offload_device=offload_device)
        self.model.eval()
        
        self.text_dim = 4096
        self.max_length = 512
        self.pad_token = 0
        self.end_token = 1
    
    def tokenize(self, text):
        """
        Tokenize text to input IDs
        
        Args:
            text: Text string or list of strings
            
        Returns:
            tuple: (input_ids, attention_mask) tensors
        """
        if isinstance(text, str):
            text = [text]
        
        if self.use_transformers:
            inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            # SentencePiece tokenizer
            input_ids_list = []
            attention_masks = []
            for t in text:
                ids = self.tokenizer.encode(t)
                # Pad or truncate
                if len(ids) < self.max_length:
                    mask = [1] * len(ids) + [0] * (self.max_length - len(ids))
                    ids = ids + [self.pad_token] * (self.max_length - len(ids))
                else:
                    ids = ids[:self.max_length]
                    mask = [1] * self.max_length
                input_ids_list.append(ids)
                attention_masks.append(mask)
            
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        
        return input_ids, attention_mask
    
    def encode_token_weights(self, token_weight_pairs):
        """
        Encode tokens with weights (ComfyUI-compatible method)
        
        This method implements the ComfyUI token weight encoding pattern:
        1. Encodes multiple token sections with weights
        2. Generates empty token encoding for blending
        3. Applies weighted blending: z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
        4. Returns (cond, pooled, extra) tuple
        
        Args:
            token_weight_pairs: List of sections, each containing [(token, weight)] pairs
            
        Returns:
            tuple: (cond, pooled, extra_dict)
                - cond: Text embeddings tensor [B, L, 4096]
                - pooled: None (T5 doesn't have pooled output)
                - extra_dict: {"attention_mask": mask_tensor}
        """
        to_encode = []
        max_token_len = 0
        has_weights = False
        
        # Process token weight pairs
        for x in token_weight_pairs:
            tokens = [a[0] for a in x]
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(a[1] == 1.0 for a in x)
            to_encode.append(tokens)
        
        sections = len(to_encode)
        
        # Add empty tokens for weighted blending
        if has_weights or sections == 0:
            empty_tokens = [self.pad_token] * max_token_len
            to_encode.append(empty_tokens)
        
        # Pad all token sequences to max length
        attention_masks = []
        for i, tokens in enumerate(to_encode):
            if len(tokens) < max_token_len:
                mask = [1] * len(tokens) + [0] * (max_token_len - len(tokens))
                tokens.extend([self.pad_token] * (max_token_len - len(tokens)))
            else:
                mask = [1] * len(tokens)
            attention_masks.append(mask)
            to_encode[i] = tokens
        
        # Convert to tensors
        input_ids = torch.tensor(to_encode, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
        
        # Encode all sections
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply weighted blending
        if has_weights:
            z_empty = out[-1]
            for k in range(sections):
                z = out[k:k+1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
        
        # Concatenate outputs
        if sections == 0:
            cond = out[-1:].float()
            final_attention_mask = attention_mask[-1:].flatten().unsqueeze(0)
        else:
            cond = torch.cat([out[k:k+1] for k in range(sections)], dim=-2).float()
            final_attention_mask = attention_mask[:sections].flatten().unsqueeze(0)
        
        # T5 doesn't have pooled output
        pooled = None
        
        # Return attention mask in extra dict
        extra = {"attention_mask": final_attention_mask}
        
        return cond, pooled, extra
    
    def encode(self, text, raw_text=False):
        """
        Encode text to embeddings [including weighting]
        Args:
            text: Text string or list of strings
            raw_text: If True, treat text as raw strings without weights
        Returns:
            Text embeddings with shape [B, L, 4096]
        """
        if raw_text:
            input_ids, attention_mask = self.tokenize(text)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Encode
            with torch.no_grad():
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return embeddings
        else:
            # Parse weighted text
            if isinstance(text, str):
                text = [text]
            token_weight_pairs_batch = []
            for t in text:
                pairs = []
                tokens = t.split(" ")
                for token in tokens:
                    if ":" in token:
                        tok, weight = token.rsplit(":", 1)
                        try:
                            weight = float(weight)
                        except ValueError:
                            tok = token
                            weight = 1.0
                    else:
                        tok = token
                        weight = 1.0
                    pairs.append((tok, weight))
                token_weight_pairs_batch.append(pairs)
            
            cond, pooled, extra = self.encode_token_weights(token_weight_pairs_batch)
            return cond
    
    def encode_bw(self, text, return_mask=False):
        """
        Simple text encoding method (backward compatibility)
        
        Args:
            text: Text string or list of strings
            return_mask: Whether to return attention mask
            
        Returns:
            Text embeddings with shape [B, L, 4096]
            If return_mask=True, returns (embeddings, attention_mask) tuple
        """
        input_ids, attention_mask = self.tokenize(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Encode
        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if return_mask:
            return embeddings, attention_mask
        return embeddings
    
    def __call__(self, text, return_mask=False):
        """Alias for encode"""
        return self.encode(text, return_mask=return_mask)
    
    @property
    def output_dim(self):
        """Get output dimension"""
        return self.text_dim
    
    def load_state_dict(self, state_dict, prefix=""):
        """Load weights from ComfyUI checkpoint"""
        return self.model.load_state_dict(state_dict, strict=False)

