"""
Sampling utilities for Wan2.2 model
Based on ComfyUI's sampling implementation

Note: Wan2.2 operates on 48-channel latents. When initializing the model for sampling,
ensure that both in_dim and out_dim are set to 48 to match the latent space dimensions.
The model predicts noise in the same channel space as its input.
"""

import torch
import numpy as np
import logging
import math
from typing import Optional, Callable, Dict, Any


def prepare_noise(latent_image: torch.Tensor, seed: int, noise_inds=None) -> torch.Tensor:
    """
    Creates random noise given a latent image and a seed.
    
    Args:
        latent_image: Reference latent tensor for shape
        seed: Random seed for noise generation
        noise_inds: Optional indices for batch noise generation
        
    Returns:
        Random noise tensor with same shape as latent_image
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(
            latent_image.size(), 
            dtype=latent_image.dtype, 
            layout=latent_image.layout, 
            generator=generator, 
            device="cpu"
        )

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn(
            [1] + list(latent_image.size())[1:], 
            dtype=latent_image.dtype, 
            layout=latent_image.layout, 
            generator=generator, 
            device="cpu"
        )
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def simple_scheduler(num_steps: int, sigma_max: float = 1.0, sigma_min: float = 0.0) -> torch.Tensor:
    """
    Simple linear scheduler for sigma values.
    
    Args:
        num_steps: Number of sampling steps
        sigma_max: Maximum sigma value (noise level)
        sigma_min: Minimum sigma value
        
    Returns:
        Tensor of sigma values for each step
    """
    sigmas = torch.linspace(sigma_max, sigma_min, num_steps + 1)
    return sigmas


def ddim_scheduler(num_steps: int, sigma_max: float = 1.0, sigma_min: float = 0.0) -> torch.Tensor:
    """
    DDIM scheduler for sigma values.
    
    Args:
        num_steps: Number of sampling steps
        sigma_max: Maximum sigma value
        sigma_min: Minimum sigma value
        
    Returns:
        Tensor of sigma values for each step
    """
    # Create evenly spaced timesteps
    timesteps = torch.linspace(0, 1, num_steps + 1)
    # Map to sigma values
    sigmas = sigma_max - timesteps * (sigma_max - sigma_min)
    return sigmas


def cosine_scheduler(num_steps: int, sigma_max: float = 1.0, sigma_min: float = 0.0) -> torch.Tensor:
    """
    Cosine scheduler for smoother sigma decay.
    
    Args:
        num_steps: Number of sampling steps
        sigma_max: Maximum sigma value
        sigma_min: Minimum sigma value
        
    Returns:
        Tensor of sigma values for each step
    """
    timesteps = torch.linspace(0, 1, num_steps + 1)
    # Cosine schedule
    alphas = torch.cos((timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
    # Convert to sigma
    sigmas = sigma_max * (1 - alphas) + sigma_min * alphas
    return sigmas


SCHEDULER_FUNCTIONS = {
    "simple": simple_scheduler,
    "ddim": ddim_scheduler,
    "cosine": cosine_scheduler,
}


def get_sigmas(scheduler_name: str, num_steps: int, sigma_max: float = 1.0, sigma_min: float = 0.0) -> torch.Tensor:
    """
    Get sigma schedule based on scheduler name.
    
    Args:
        scheduler_name: Name of the scheduler ('simple', 'ddim', 'cosine')
        num_steps: Number of sampling steps
        sigma_max: Maximum sigma value
        sigma_min: Minimum sigma value
        
    Returns:
        Tensor of sigma values
    """
    if scheduler_name not in SCHEDULER_FUNCTIONS:
        logging.warning(f"Unknown scheduler '{scheduler_name}', using 'simple'")
        scheduler_name = "simple"
    
    scheduler_fn = SCHEDULER_FUNCTIONS[scheduler_name]
    return scheduler_fn(num_steps, sigma_max, sigma_min)


def cfg_function(
    model_output_cond: torch.Tensor,
    model_output_uncond: torch.Tensor,
    cfg_scale: float,
    x: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Apply classifier-free guidance (CFG) to model outputs.
    
    Args:
        model_output_cond: Model output with conditioning
        model_output_uncond: Model output without conditioning (unconditional)
        cfg_scale: CFG scale (typically 7.0-15.0)
        x: Current noisy latent
        sigma: Current sigma value
        
    Returns:
        CFG-guided model output
    """
    # Standard CFG formula
    return model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)


def euler_step(
    x: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor
) -> torch.Tensor:
    """
    Single Euler sampling step.
    
    Args:
        x: Current noisy latent
        model_output: Model's denoised prediction
        sigma: Current sigma value
        sigma_next: Next sigma value
        
    Returns:
        Updated latent for next step
    """
    # Euler integration step
    d = (x - model_output) / sigma
    dt = sigma_next - sigma
    return x + d * dt


def euler_ancestral_step(
    x: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 1.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Single Euler ancestral sampling step (adds noise).
    
    Args:
        x: Current noisy latent
        model_output: Model's denoised prediction
        sigma: Current sigma value
        sigma_next: Next sigma value
        eta: Stochasticity parameter (0 = deterministic, 1 = fully stochastic)
        seed: Random seed for noise
        
    Returns:
        Updated latent for next step
    """
    # Calculate noise amount
    sigma_up = torch.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2) * eta
    sigma_down = torch.sqrt(sigma_next**2 - sigma_up**2)
    
    # Euler step
    d = (x - model_output) / sigma
    dt = sigma_down - sigma
    x = x + d * dt
    
    # Add noise if not at final step
    if sigma_next > 0:
        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None
        noise = torch.randn_like(x, generator=generator)
        x = x + noise * sigma_up
    
    return x


def ddim_step(
    x: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Single DDIM sampling step.
    
    Args:
        x: Current noisy latent
        model_output: Model's denoised prediction
        sigma: Current sigma value (noise level)
        sigma_next: Next sigma value
        eta: Stochasticity parameter (0 = deterministic DDIM)
        
    Returns:
        Updated latent for next step
    """
    # DDIM update rule
    # Convert sigma to alpha
    alpha = 1 / (1 + sigma**2)
    alpha_next = 1 / (1 + sigma_next**2)
    
    # Predict x0 (clean image)
    pred_x0 = model_output
    
    # Add noise component if eta > 0
    if eta > 0 and sigma_next > 0:
        noise = torch.randn_like(x)
        sigma_noise = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
    else:
        noise = 0
        sigma_noise = 0
    
    # DDIM update
    x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next - sigma_noise**2) * ((x - torch.sqrt(alpha) * pred_x0) / torch.sqrt(1 - alpha)) + sigma_noise * noise
    
    return x


class SimpleSampler:
    """
    Simple sampler for Wan2.2 model.
    Implements basic sampling algorithms like Euler and DDIM.
    """
    
    SAMPLERS = ["euler", "euler_ancestral", "ddim"]
    SCHEDULERS = list(SCHEDULER_FUNCTIONS.keys())
    
    def __init__(
        self,
        model: torch.nn.Module,
        steps: int = 50,
        sampler: str = "euler",
        scheduler: str = "simple",
        cfg_scale: float = 7.0,
        denoise: float = 1.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0
    ):
        """
        Initialize sampler.
        
        Args:
            model: Wan2.2 model
            steps: Number of sampling steps
            sampler: Sampler type ('euler', 'euler_ancestral', 'ddim')
            scheduler: Scheduler type ('simple', 'ddim', 'cosine')
            cfg_scale: Classifier-free guidance scale
            denoise: Denoising strength (1.0 = full denoising)
            sigma_max: Maximum noise level
            sigma_min: Minimum noise level
        """
        self.model = model
        self.steps = steps
        self.sampler = sampler if sampler in self.SAMPLERS else "euler"
        self.scheduler = scheduler if scheduler in self.SCHEDULERS else "simple"
        self.cfg_scale = cfg_scale
        self.denoise = denoise
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        
        # Calculate sigmas
        self.sigmas = self.calculate_sigmas()
    
    def calculate_sigmas(self) -> torch.Tensor:
        """Calculate sigma schedule."""
        steps = self.steps
        if self.denoise < 1.0:
            # Adjust steps for partial denoising
            new_steps = int(steps / self.denoise)
            sigmas = get_sigmas(self.scheduler, new_steps, self.sigma_max, self.sigma_min)
            sigmas = sigmas[-(steps + 1):]
        else:
            sigmas = get_sigmas(self.scheduler, steps, self.sigma_max, self.sigma_min)
        return sigmas
    
    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        positive_conditioning: torch.Tensor,
        negative_conditioning: Optional[torch.Tensor] = None,
        latent_image: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        disable_pbar: bool = False,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Run sampling loop.
        
        Args:
            noise: Initial noise tensor [B, C, T, H, W]
            positive_conditioning: Text embeddings for positive prompt [B, L, D]
            negative_conditioning: Text embeddings for negative prompt [B, L, D] (optional)
            latent_image: Starting latent for img2img (optional)
            callback: Callback function called each step with (step, x)
            disable_pbar: Whether to disable progress bar
            seed: Random seed
            
        Returns:
            Denoised latent tensor [B, C, T, H, W]
        """
        device = noise.device
        dtype = noise.dtype
        
        # Move sigmas to device
        sigmas = self.sigmas.to(device)
        
        # Initialize from noise or latent_image
        if latent_image is not None:
            # Add noise to latent_image based on starting sigma
            x = latent_image + noise * sigmas[0]
        else:
            x = noise * sigmas[0]
        
        # Use classifier-free guidance if negative conditioning provided
        use_cfg = negative_conditioning is not None and self.cfg_scale > 1.0
        
        # Sampling loop
        total_steps = len(sigmas) - 1
        
        if not disable_pbar:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(total_steps), desc="Sampling")
            except ImportError:
                iterator = range(total_steps)
        else:
            iterator = range(total_steps)
        
        for i in iterator:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Prepare timestep (scale sigma to timestep range)
            timestep = sigma.unsqueeze(0)
            
            # Model forward pass with conditioning
            if use_cfg:
                # Run model with both positive and negative conditioning
                # Concatenate for batched forward pass
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([timestep, timestep], dim=0)
                context_in = torch.cat([positive_conditioning, negative_conditioning], dim=0)
                
                model_output = self.model(x_in, t_in, context_in)
                
                # Split outputs
                model_output_cond, model_output_uncond = model_output.chunk(2, dim=0)
                
                # Apply CFG
                model_output = cfg_function(
                    model_output_cond, 
                    model_output_uncond, 
                    self.cfg_scale,
                    x,
                    sigma
                )
            else:
                # Run model with only positive conditioning
                model_output = self.model(x, timestep, positive_conditioning)
            
            # Apply sampling step based on sampler type
            if self.sampler == "euler":
                x = euler_step(x, model_output, sigma, sigma_next)
            elif self.sampler == "euler_ancestral":
                step_seed = (seed + i) if seed is not None else None
                x = euler_ancestral_step(x, model_output, sigma, sigma_next, eta=1.0, seed=step_seed)
            elif self.sampler == "ddim":
                x = ddim_step(x, model_output, sigma, sigma_next, eta=0.0)
            
            # Call callback if provided
            if callback is not None:
                callback(i, x)
        
        return x
    
    def sample_simple(
        self,
        shape: tuple,
        positive_conditioning: torch.Tensor,
        negative_conditioning: Optional[torch.Tensor] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None
    ) -> torch.Tensor:
        """
        Simplified sampling interface that handles noise generation.
        
        Args:
            shape: Shape of latent to generate [B, C, T, H, W]
            positive_conditioning: Text embeddings for positive prompt
            negative_conditioning: Text embeddings for negative prompt (optional)
            device: Device to run on
            dtype: Data type
            seed: Random seed
            callback: Progress callback
            
        Returns:
            Denoised latent tensor
        """
        # Generate noise
        noise = prepare_noise(
            torch.zeros(shape, device=device, dtype=dtype),
            seed
        ).to(device).to(dtype)
        
        # Run sampling
        return self.sample(
            noise=noise,
            positive_conditioning=positive_conditioning,
            negative_conditioning=negative_conditioning,
            callback=callback,
            seed=seed
        )


def sample(
    model: torch.nn.Module,
    noise: torch.Tensor,
    positive: torch.Tensor,
    negative: Optional[torch.Tensor],
    cfg_scale: float,
    steps: int = 50,
    sampler: str = "euler",
    scheduler: str = "simple",
    denoise: float = 1.0,
    latent_image: Optional[torch.Tensor] = None,
    callback: Optional[Callable] = None,
    disable_pbar: bool = False,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    High-level sampling function similar to ComfyUI's sample().
    
    Args:
        model: Wan2.2 model
        noise: Initial noise
        positive: Positive text conditioning
        negative: Negative text conditioning
        cfg_scale: CFG scale
        steps: Number of steps
        sampler: Sampler name
        scheduler: Scheduler name
        denoise: Denoising strength
        latent_image: Optional starting latent
        callback: Progress callback
        disable_pbar: Disable progress bar
        seed: Random seed
        
    Returns:
        Denoised latent
    """
    sampler_obj = SimpleSampler(
        model=model,
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        cfg_scale=cfg_scale,
        denoise=denoise
    )
    
    return sampler_obj.sample(
        noise=noise,
        positive_conditioning=positive,
        negative_conditioning=negative,
        latent_image=latent_image,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )
