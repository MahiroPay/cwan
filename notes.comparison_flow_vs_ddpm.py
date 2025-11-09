"""
Comparison: Flow Matching vs DDPM Training for Wan2.2

This file shows the key differences between flow matching and traditional
diffusion model (DDPM) training approaches.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# FLOW MATCHING TRAINING (Recommended for Wan2.2)
# =============================================================================

def flow_matching_training_step(model, x_1, context, optimizer):
    """
    Flow Matching Training Step
    
    Simpler, more stable, and theoretically grounded in optimal transport.
    """
    batch_size = x_1.shape[0]
    
    # Sample uniform timesteps in [0, 1]
    t = torch.rand(batch_size, device=x_1.device)
    
    # Sample source noise
    x_0 = torch.randn_like(x_1)
    
    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    t_expanded = t.view(-1, 1, 1, 1, 1)
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
    
    # Target is the velocity: v = x_1 - x_0
    target_velocity = x_1 - x_0
    
    # Model predicts velocity
    predicted_velocity = model(x_t, t * 1000, context)
    
    # Simple MSE loss
    loss = F.mse_loss(predicted_velocity, target_velocity)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


# =============================================================================
# DDPM TRAINING (Traditional Diffusion)
# =============================================================================

def ddpm_training_step(model, x_0_data, context, optimizer, alphas_cumprod):
    """
    DDPM Training Step
    
    More complex with noise schedules, but well-established approach.
    """
    batch_size = x_0_data.shape[0]
    
    # Sample random timesteps from discrete schedule [0, T-1]
    T = 1000
    t = torch.randint(0, T, (batch_size,), device=x_0_data.device)
    
    # Get noise schedule values
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    
    # Sample noise
    epsilon = torch.randn_like(x_0_data)
    
    # Noisy sample: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
    x_t = torch.sqrt(alpha_t) * x_0_data + torch.sqrt(1 - alpha_t) * epsilon
    
    # Model predicts the noise
    predicted_noise = model(x_t, t, context)
    
    # MSE loss on noise prediction
    loss = F.mse_loss(predicted_noise, epsilon)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


# =============================================================================
# V-PREDICTION TRAINING (Alternative Diffusion)
# =============================================================================

def v_prediction_training_step(model, x_0_data, context, optimizer, alphas_cumprod):
    """
    V-Prediction Training Step
    
    Predicts v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x_0
    Can be more stable than pure noise prediction.
    """
    batch_size = x_0_data.shape[0]
    T = 1000
    
    t = torch.randint(0, T, (batch_size,), device=x_0_data.device)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    
    epsilon = torch.randn_like(x_0_data)
    
    # Noisy sample
    x_t = torch.sqrt(alpha_t) * x_0_data + torch.sqrt(1 - alpha_t) * epsilon
    
    # Target v-prediction
    target_v = torch.sqrt(alpha_t) * epsilon - torch.sqrt(1 - alpha_t) * x_0_data
    
    # Model predicts v
    predicted_v = model(x_t, t, context)
    
    loss = F.mse_loss(predicted_v, target_v)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================

"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FLOW MATCHING vs DDPM                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FLOW MATCHING                        │  DDPM                                │
│  ──────────────                       │  ────                                │
│                                       │                                      │
│  Time:                                │  Time:                               │
│    t ~ Uniform[0, 1]                  │    t ~ Uniform[0, ..., T-1]          │
│    Simple, continuous                 │    Discrete, needs schedule          │
│                                       │                                      │
│  Noise Schedule:                      │  Noise Schedule:                     │
│    None needed!                       │    α_t = ∏(1 - β_i)                  │
│    Just linear interpolation          │    β_1, β_2, ..., β_T                │
│                                       │    Requires careful tuning           │
│                                       │                                      │
│  Noisy Sample:                        │  Noisy Sample:                       │
│    x_t = (1-t)·x_0 + t·x_1            │   x_t = √α_t·x_0 + √(1-α_t)·ε        │
│    Linear path                        │    Non-linear, schedule-dependent    │
│                                       │                                      │
│  Model Output:                        │  Model Output:                       │
│    Velocity: v = x_1 - x_0            │    Noise: ε                          │
│    Direction to data                  │    Pure noise prediction             │
│                                       │                                      │
│  Loss:                                │  Loss:                               │
│    ||model(x_t,t,c) - (x_1-x_0)||²    │    ||model(x_t,t,c) - ε||²           │
│    Simple MSE                         │    Simple MSE                        │
│                                       │                                      │
│  Sampling:                            │  Sampling:                           │
│    ODE: x_{t+dt} = x_t + v·dt         │    Iterative: x_{t-1} from x_t       │
│    Deterministic, fast                │    Can be stochastic                 │
│    20-50 steps typical                │    50-1000 steps typical             │
│                                       │                                      │
│  Advantages:                          │  Advantages:                         │
│    ✓ No noise schedule                │    ✓ Well-established               │
│    ✓ Simpler code                     │    ✓ Lots of research               │
│    ✓ Faster sampling                  │    ✓ Many variants (DDIM, PNDM)     │
│    ✓ Theoretical guarantees           │    ✓ Proven at scale                │
│    ✓ Straighter paths                 │                                     │
│                                       │                                      │
│  Disadvantages:                       │  Disadvantages:                      │
│    ✗ Less research/tooling            │    ✗ Complex noise schedule         │
│    ✗ Newer approach                   │    ✗ More hyperparameters           │
│                                       │    ✗ Slower sampling                │
│                                       │    ✗ Curved paths in latent space   │
│                                       │                                      │
└──────────────────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# VISUAL COMPARISON OF TRAJECTORIES
# =============================================================================

"""
Flow Matching Path (Linear):
────────────────────────────

    x₁ (data)
     ↑
     │ v = x₁ - x₀
     │  (straight line)
     │
     │
     ↑
    x_t
     │
     │
     ↑
    x₀ (noise)

t: 0 ──────────────────────> 1


DDPM Path (Curved):
───────────────────

    x₀ (data)
     ↑
     │╲
     │ ╲  curved path
     │  ╲ due to noise
     │   ╲ schedule
     │    ╲
     │     ↑
     │    x_t
     │   ╱
     │  ╱
     │ ╱
     │╱
     ↑
    x_T (noise)

t: T ──────────────────────> 0
"""


# =============================================================================
# WHICH ONE TO USE FOR WAN2.2?
# =============================================================================

"""
RECOMMENDATION: Flow Matching

Why?
────

1. Wan2.2 is a modern architecture designed for flow-based training
   (similar to Stable Video Diffusion, Stable Diffusion 3, etc.)

2. Simpler to implement - no noise schedule hyperparameters

3. Faster sampling - important for video generation (fewer steps)

4. Better quality - straighter paths through 48-channel latent space

5. More stable training - no need to balance noise schedule

6. Current state-of-the-art video models use flow matching:
   - Stable Video Diffusion
   - Sora (likely)
   - Lumiere
   - VideoPoet


When might you use DDPM instead?
─────────────────────────────────

- You have a well-tuned DDPM noise schedule already
- You're fine-tuning a pre-trained DDPM model
- You need to match existing DDPM-based inference code
- You have specific requirements for stochastic sampling


Training Time Comparison:
─────────────────────────

Flow Matching:
  - Epochs to converge: ~100-200
  - Sampling steps: 20-50
  - Training stability: High
  
DDPM:
  - Epochs to converge: ~100-300
  - Sampling steps: 50-1000 (can use DDIM for faster)
  - Training stability: Medium (schedule dependent)


Code Complexity:
────────────────

Flow Matching: ~50 lines core code
DDPM: ~150 lines (including schedule logic)
"""


# =============================================================================
# EXAMPLE: MINIMAL IMPLEMENTATIONS
# =============================================================================




def minimal_flow_matching(B, x1, model, c):
    """Absolute minimal flow matching - fits in a tweet!"""
    t = torch.rand(B, 1, 1, 1, 1)
    x0 = torch.randn_like(x1)
    xt = (1-t)*x0 + t*x1
    loss = F.mse_loss(model(xt, t*1000, c), x1 - x0)


def minimal_ddpm(B, T, x0, model, c, sqrt_alpha, sqrt_one_minus_alpha):
    """Absolute minimal DDPM - needs noise schedule"""
    t = torch.randint(0, T, (B,))
    eps = torch.randn_like(x0)
    xt = sqrt_alpha[t]*x0 + sqrt_one_minus_alpha[t]*eps
    loss = F.mse_loss(model(xt, t, c), eps)


# =============================================================================
# SAMPLING COMPARISON
# =============================================================================

def sample_flow_matching(model, context, steps=50, T=16, H=64, W=64):
    """Flow matching sampling: ODE integration"""
    x = torch.randn(1, 48, T, H, W)
    dt = 1.0 / steps
    
    for i in range(steps):
        t = i / steps
        v = model(x, t * 1000, context)
        x = x + v * dt  # Euler step
    
    return x

from numpy import sqrt

def sample_ddpm(model, context, steps=50, T=16, H=64, W=64, alpha_schedule=None):
    """DDPM sampling: iterative denoising"""
    x = torch.randn(1, 48, T, H, W)
    
    for i in reversed(range(steps)):
        t = torch.tensor([i])
        
        # Predict noise
        eps = model(x, t, context)
        
        # Compute x_{t-1} from x_t
        alpha_t = alpha_schedule[i]
        alpha_prev = alpha_schedule[i-1] if i > 0 else 1.0
        
        # DDIM step (deterministic)
        pred_x0 = (x - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        direction = sqrt(1 - alpha_prev) * eps
        x = sqrt(alpha_prev) * pred_x0 + direction
    
    return x

