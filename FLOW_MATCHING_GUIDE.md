# Flow Matching Training for Wan2.2

This guide explains how to train Wan2.2 using flow matching (also called rectified flow).

## Overview

Flow matching is a modern approach to generative modeling that learns to transport noise to data through optimal paths. It's simpler and often more effective than traditional diffusion models (DDPM/DDIM).

## Core Concept

### The Flow Equation

Given:
- **x₀** ~ N(0, I) : Pure Gaussian noise
- **x₁** ~ p_data : Real data (latent)
- **t** ∈ [0, 1] : Time parameter

The interpolation path is:
```
x_t = (1 - t) * x₀ + t * x₁
```

The velocity field to learn:
```
v = dx/dt = x₁ - x₀
```

### Training Objective

The model learns to predict the velocity field:
```
L = E_[x₀, x₁, t] [ ||v_θ(x_t, t, c) - (x₁ - x₀)||² ]
```

Where:
- `v_θ` is the neural network (Wan2.2 model)
- `c` is the conditioning (text embeddings)
- `t` is sampled uniformly from [0, 1]

## Training Algorithm

### Pseudocode

```python
for batch in dataloader:
    # 1. Get clean data
    x_1 = encode_video(batch['video'])  # Shape: [B, 48, T, H, W]
    context = encode_text(batch['text'])  # Shape: [B, L, 4096]
    
    # 2. Sample random timesteps
    t = torch.rand(batch_size)  # Shape: [B]
    
    # 3. Sample noise
    x_0 = torch.randn_like(x_1)  # Shape: [B, 48, T, H, W]
    
    # 4. Create noisy sample via interpolation
    t_expanded = t.view(B, 1, 1, 1, 1)
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
    
    # 5. Compute target velocity
    target = x_1 - x_0  # The direction from noise to data
    
    # 6. Model prediction
    predicted = model(x_t, t * 1000, context)
    
    # 7. Compute loss and optimize
    loss = mse_loss(predicted, target)
    loss.backward()
    optimizer.step()
```

## Wan2.2 Specifics

### Model Configuration

```python
model = WanModel(
    model_type='t2v',        # Text-to-video
    patch_size=(1, 2, 2),    # Temporal, height, width patches
    in_dim=48,               # 48-channel latents (Wan2.2 specific)
    out_dim=48,              # Predict in same space
    dim=2048,                # Hidden dimension
    ffn_dim=8192,            # FFN dimension
    num_heads=16,            # Attention heads
    num_layers=32,           # Transformer layers
    text_len=512,            # Max text length
    text_dim=4096,           # Text embedding dimension (T5-XXL)
)
```

### Latent Space

Wan2.2 operates on **48-channel latents** encoded by the VAE:
- Input video: `[B, 3, T, H, W]` (RGB video)
- VAE encoding: `[B, 16, T', H', W']` → normalize → `[B, 48, T', H', W']`
- Spatial compression: 8x (height and width)
- Temporal compression: variable (typically 4x)

### Timestep Handling

The Wan2.2 model expects timesteps in range [0, 1000]:
```python
# Flow matching uses t in [0, 1]
t = torch.rand(batch_size)

# Scale for Wan2.2
timestep_scaled = t * 1000.0

# Pass to model
output = model(x_t, timestep_scaled, context)
```

## Complete Training Loop

See `train_flow_matching.py` for the full implementation with:
- EMA (Exponential Moving Average) for better sampling
- Gradient clipping
- Learning rate scheduling
- Checkpointing
- Metrics logging

For a minimal example, see `traintest.py`.

## Sampling (Inference)

After training, generate videos using ODE solvers:

### Euler Method (Simplest)

```python
def sample_euler(model, context, num_steps=50):
    # Start from pure noise
    x = torch.randn(1, 48, T, H, W).cuda()
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        
        # Predict velocity
        with torch.no_grad():
            v = model(x, t * 1000, context)
        
        # Euler step: x = x + v * dt
        x = x + v * dt
    
    return x
```

### Heun's Method (Better)

```python
def sample_heun(model, context, num_steps=50):
    x = torch.randn(1, 48, T, H, W).cuda()
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        
        with torch.no_grad():
            # First prediction
            v1 = model(x, t * 1000, context)
            x_temp = x + v1 * dt
            
            # Second prediction
            v2 = model(x_temp, (t + dt) * 1000, context)
            
            # Average
            v = (v1 + v2) / 2
            x = x + v * dt
    
    return x
```

## Key Differences from DDPM

| Aspect | DDPM/DDIM | Flow Matching |
|--------|-----------|---------------|
| **Time range** | [0, T] with noise schedule | [0, 1] uniform |
| **Target** | Predict noise ε | Predict velocity v |
| **Path** | Markov chain | Continuous flow |
| **Sampling** | Iterative denoising | ODE integration |
| **Theory** | Score matching | Optimal transport |
| **Complexity** | Noise schedule, variance | Simple interpolation |

## Advantages of Flow Matching

1. **Simpler**: No noise schedule to tune
2. **Faster**: Fewer sampling steps needed
3. **More stable**: Straighter paths through latent space
4. **Theoretically grounded**: Based on optimal transport
5. **Better quality**: Often produces higher quality samples
6. **Flexible**: Easy to change paths (rectified flow)

## Tips for Training

### Learning Rate
- Start: 1e-4
- Warmup: 1000-5000 steps
- Schedule: Cosine annealing
- Min LR: 1e-6

### Batch Size
- Depends on GPU memory
- Wan2.2 is memory-intensive due to 48 channels
- Typical: 1-4 videos per GPU (A100 80GB)
- Use gradient accumulation if needed

### Gradient Clipping
- Clip norm: 1.0
- Prevents instability

### EMA
- Decay: 0.9999
- Use EMA model for inference
- Better sample quality

### Data Augmentation
- Temporal cropping
- Spatial cropping/resizing
- Color jittering (mild)
- Avoid heavy augmentation (hurts quality)

### Mixed Precision
- Use bfloat16 on A100/H100
- Use float16 on V100 (with careful tuning)
- Keep sensitive ops in float32

## Monitoring

Track these metrics during training:

1. **Loss**: Should decrease steadily
2. **Velocity magnitude**: `||predicted||` and `||target||` should be similar
3. **Cosine similarity**: Between predicted and target velocity (should increase)
4. **Gradient norm**: Should be stable, not exploding
5. **Sample quality**: Generate samples periodically

## Troubleshooting

### Loss not decreasing
- Check learning rate (might be too high/low)
- Verify data preprocessing
- Check gradient flow

### NaN/Inf values
- Reduce learning rate
- Enable gradient clipping
- Check for numerical instability in data

### Poor sample quality
- Train longer
- Use EMA model
- Try different ODE solvers
- Increase sampling steps

### Memory issues
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training
- Reduce sequence length

## References

1. **Flow Matching for Generative Modeling**
   - Lipman et al., ICLR 2023
   - Introduces flow matching framework

2. **Flow Straight and Fast**
   - Liu et al., 2023 (Rectified Flow)
   - Improves path straightness

3. **Stable Video Diffusion**
   - Blattmann et al., 2023
   - Video generation with flow matching

4. **Wan2.1 & Wan2.2 Papers**
   - Alibaba Wan Team
   - Original architecture details

## Example Usage

```python
from train_flow_matching import FlowMatchingTrainer
from comfywan import WanModel, WanVAE, WanT5TextEncoder

# Initialize
model = WanModel(in_dim=48, out_dim=48, ...)
vae = WanVAE()
text_encoder = WanT5TextEncoder()

# Create trainer
trainer = FlowMatchingTrainer(
    model=model,
    vae=vae,
    text_encoder=text_encoder,
    learning_rate=1e-4,
    device='cuda',
    dtype=torch.bfloat16,
)

# Train
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(dataloader, scheduler)
    if epoch % 10 == 0:
        trainer.save_checkpoint(f'checkpoint_{epoch}.pt')
```

## Next Steps

1. Prepare your video dataset
2. Set up VAE and text encoder
3. Configure training hyperparameters
4. Run training with `train_flow_matching.py`
5. Monitor metrics and samples
6. Evaluate and iterate

Good luck with your training! 🚀
