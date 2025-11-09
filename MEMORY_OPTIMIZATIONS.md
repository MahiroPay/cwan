# Memory Optimization Guide

This document describes the memory optimization features available in the cwan repository for training and inference with the Wan2.2 video diffusion model.

## Overview

Memory optimizations are crucial when working with large models like Wan2.2 (5B parameters). This implementation provides two key memory optimization techniques:

1. **Gradient Checkpointing** - For memory-efficient training
2. **Block Offloading** - For memory-efficient inference

## Gradient Checkpointing

Gradient checkpointing trades computation for memory by not storing intermediate activations during the forward pass. Instead, activations are recomputed during the backward pass when needed. This can significantly reduce memory usage during training at the cost of ~30-40% slower training speed.

### How It Works

- During forward pass: Only checkpoint boundaries are saved
- During backward pass: Intermediate activations are recomputed on-the-fly
- Memory savings: Up to 50-70% reduction in activation memory
- Performance impact: ~30-40% slower training

### Usage in Training

Enable gradient checkpointing when training the Wan2.2 model:

```bash
python train_flow_matching.py \
    --dataset_path ./dataset/ \
    --batch_size 1 \
    --num_epochs 100 \
    --gradient_checkpointing
```

If you're also training with a VAE, enable VAE gradient checkpointing:

```bash
python train_flow_matching.py \
    --dataset_path ./dataset/ \
    --batch_size 1 \
    --num_epochs 100 \
    --gradient_checkpointing \
    --vae_gradient_checkpointing
```

### Programmatic API

You can also enable/disable gradient checkpointing programmatically:

```python
from comfywan import WanModel

# Create model with gradient checkpointing enabled
model = WanModel(
    model_type='t2v',
    num_layers=30,
    gradient_checkpointing=True,
    # ... other parameters
)

# Or enable/disable dynamically
model.enable_gradient_checkpointing()
model.disable_gradient_checkpointing()
```

For the VAE:

```python
from comfywan import WanVAE

# Create VAE with gradient checkpointing enabled
vae = WanVAE(
    z_dim=16,
    gradient_checkpointing=True,
    # ... other parameters
)

# Or enable/disable dynamically
vae.enable_gradient_checkpointing()
vae.disable_gradient_checkpointing()
```

## Block Offloading

Block offloading moves transformer blocks to CPU memory when they're not actively being used. During inference, only the current block being executed is kept on GPU, while others are on CPU. This allows running very large models on GPUs with limited VRAM.

### How It Works

- Each transformer block is moved to GPU only when needed
- After processing, the block is moved back to CPU
- GPU memory is freed after each block
- Sequential processing ensures minimal GPU memory usage

### Performance Trade-offs

- Memory savings: Proportional to number of blocks (e.g., 30 blocks = ~30x less GPU memory for model weights)
- Performance impact: ~2-3x slower inference due to CPU↔GPU transfers
- Best for: Inference when GPU memory is very limited

### Usage in Inference

Enable block offloading for memory-efficient inference:

```bash
python test_inference.py \
    --checkpoint models/wan2.2_ti2v_5B_fp16.safetensors \
    --cond embeddings/cond.safetensors \
    --uncond embeddings/uncond.safetensors \
    --steps 30 \
    --offload-to-cpu
```

### Programmatic API

```python
from comfywan import WanModel

# Create model with offloading enabled
model = WanModel(
    model_type='t2v',
    num_layers=30,
    offload_to_cpu=True,
    # ... other parameters
)

# Or enable/disable dynamically
model.enable_offloading()
model.disable_offloading()
```

## Combining Optimizations

You can combine both optimizations for maximum memory efficiency during training:

```bash
python train_flow_matching.py \
    --dataset_path ./dataset/ \
    --batch_size 1 \
    --num_epochs 100 \
    --gradient_checkpointing
```

**Note:** Block offloading is primarily designed for inference and is not recommended during training, as the memory savings are already handled by gradient checkpointing.

## Memory Usage Comparison

Here's an approximate comparison of GPU memory usage (for Wan2.2 5B model):

| Configuration | Model Weights | Activations | Total* | Speed |
|--------------|---------------|-------------|--------|-------|
| Baseline | ~10 GB | ~20-40 GB | ~30-50 GB | 1.0x |
| + Gradient Checkpointing | ~10 GB | ~5-10 GB | ~15-20 GB | 0.6-0.7x |
| + Block Offloading (Inference) | ~0.3 GB | ~10 GB | ~10-11 GB | 0.3-0.5x |

*Total includes batch data and optimizer states where applicable

## Best Practices

### Training

1. **Always use gradient checkpointing** when GPU memory is limited
2. **Start with smaller batch sizes** and increase if memory allows
3. **Monitor GPU memory usage** during training with `nvidia-smi`
4. **Use mixed precision (bfloat16/float16)** for additional memory savings
5. **Note:** Optimization flags are not saved in checkpoints - you must specify them when loading

### Inference

1. **Use block offloading** only if you cannot fit the model in GPU memory
2. **Prefer larger batch sizes without offloading** for better throughput
3. **Consider quantization** (e.g., int8) for additional memory savings
4. **Profile your setup** to find the optimal configuration
5. **Note:** Offloading flags must be set when loading the model

## Troubleshooting

### Out of Memory (OOM) Errors

If you still get OOM errors with optimizations enabled:

1. **Reduce batch size** to 1 (or use micro-batching)
2. **Reduce sequence length** or spatial resolution
3. **Enable both gradient checkpointing and mixed precision**
4. **For inference, enable block offloading**
5. **Consider using a machine with more VRAM**

### Slow Training

If training is too slow with gradient checkpointing:

1. **Disable gradient checkpointing** if you have enough VRAM
2. **Use gradient accumulation** to simulate larger batch sizes
3. **Profile your code** to identify other bottlenecks
4. **Consider using multiple GPUs** with data parallelism

### Slow Inference

If inference is too slow with block offloading:

1. **Disable block offloading** if you have enough VRAM
2. **Use model quantization** instead for memory savings
3. **Consider using a more powerful GPU** with more VRAM
4. **Reduce the number of sampling steps** if quality allows

## Technical Details

### Implementation Notes

- **Gradient Checkpointing**: Implemented using PyTorch's `torch.utils.checkpoint.checkpoint` with `use_reentrant=False` for compatibility
- **Block Offloading**: Blocks are moved with `.cpu()` and `.to(device)`, with explicit `torch.cuda.empty_cache()` calls
- **Thread Safety**: Not guaranteed for concurrent access with offloading enabled

### Compatibility

- **PyTorch Version**: Requires PyTorch 2.0 or later
- **CUDA**: Block offloading requires CUDA-capable GPU
- **Distributed Training**: Compatible with gradient checkpointing, block offloading not recommended

## References

- [PyTorch Gradient Checkpointing Documentation](https://pytorch.org/docs/stable/checkpoint.html)
- [Memory-Efficient Training Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- Wan2.2 Model: [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
