"""
Flow Matching Training Loop for Wan2.2

This implements a flow matching training approach for the Wan2.2 video diffusion model.
Flow matching learns a vector field that transports noise to data through optimal transport paths.

Key concepts:
- Flow matching uses conditional probability paths from noise (t=0) to data (t=1)
- The model learns to predict the velocity field v_t(x_t | x_1)
- Training objective: E[||v_theta(x_t, t, c) - (x_1 - x_0)||^2]
- Simple linear interpolation: x_t = (1-t)*x_0 + t*x_1 where x_0~N(0,I), x_1~data

Reference:
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Stable Video Diffusion" uses rectified flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm
from PIL import Image
import os
import safetensors.torch as safe_torch
from comfywan import WanModel, WanVAE, Wan22LatentFormat


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowMatchingTrainer:
    """
    Flow Matching Trainer for Wan2.2
    
    Implements rectified flow / flow matching training where:
    - x_0 ~ N(0, I) is pure noise
    - x_1 is the real data (latent)
    - x_t = (1-t)*x_0 + t*x_1 for t in [0, 1]
    - Model predicts v_t = x_1 - x_0 (the velocity/direction)
    """
    
    def __init__(
        self,
        model: WanModel,
        vae: Optional[WanVAE] = None,
        latent_format: Optional[Wan22LatentFormat] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        grad_clip: float = 1.0,
        ema_decay = 0.9999,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the flow matching trainer
        
        Args:
            model: The Wan2.2 diffusion backbone
            vae: VAE for encoding videos (optional if training on pre-encoded latents)
            latent_format: Latent normalization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW
            betas: Adam beta parameters
            grad_clip: Gradient clipping threshold
            ema_decay: Exponential moving average decay for model weights
            device: Device to train on
            dtype: Training dtype (bfloat16 recommended for A100/H100)
        """
        self.device = device
        self.dtype = dtype
        self.grad_clip = grad_clip
        
        # Model components
        self.model = model.to(device)
        self.vae = vae.to(device) if vae is not None else None
        self.latent_format = latent_format if latent_format is not None else Wan22LatentFormat()
        
        # Freeze VAE (only train the diffusion model)
        if self.vae is not None:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
                
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.ema_decay = ema_decay
        self.ema_model = None
        # EMA model for better sampling quality
        if ema_decay is not None:
            self.ema_model = self._create_ema_model()
            self.ema_decay = ema_decay
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def _create_ema_model(self):
        """Create EMA version of the model"""
        ema_model = type(self.model)(
            **{k: v for k, v in self.model.__dict__.items() 
               if not k.startswith('_') and k not in ['training']}
        ).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA model parameters"""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), 
            self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - self.ema_decay
            )
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space
        
        Args:
            video: Video tensor [B, C, T, H, W] in range [-1, 1]
            
        Returns:
            Normalized latent [B, 48, T', H', W']
        """
        with torch.no_grad():
            latent = self.vae.encode(video)
            latent = self.latent_format.process_in(latent)
        return latent
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps for flow matching
        
        Args:
            batch_size: Number of timesteps to sample
            
        Returns:
            Timesteps in [0, 1]
        """
        # Uniform sampling in [0, 1]
        # Note: Some implementations use logit-normal sampling for better coverage
        return torch.rand(batch_size, device=self.device)
    
    def get_noisy_latent(
        self, 
        x_1: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create noisy latent via linear interpolation
        
        Flow matching interpolant: x_t = (1-t)*x_0 + t*x_1
        where x_0 ~ N(0, I) is pure noise and x_1 is the data
        
        Args:
            x_1: Clean data latent [B, C, T, H, W]
            t: Timesteps [B] in [0, 1]
            
        Returns:
            x_t: Interpolated latent [B, C, T, H, W]
            x_0: Source noise [B, C, T, H, W]
        """
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t_broadcast = t.view(-1, 1, 1, 1, 1)
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        
        return x_t, x_0
    
    def compute_flow_matching_loss(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
        x_0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss
        
        The model predicts the velocity v_t = dx/dt
        Target velocity: v_t = d/dt[(1-t)*x_0 + t*x_1] = x_1 - x_0
        
        Loss: E[||v_theta(x_t, t, c) - (x_1 - x_0)||^2]
        
        Args:
            x_1: Clean data latent [B, C, T, H, W]
            t: Timesteps [B] in [0, 1]
            context: Text conditioning [B, L, D]
            x_t: Pre-computed noisy latent (optional)
            x_0: Pre-computed noise (optional)
            
        Returns:
            Dictionary with loss and metrics
        """
        # Get noisy latent if not provided
        if x_t is None or x_0 is None:
            x_t, x_0 = self.get_noisy_latent(x_1, t)
        
        # Target velocity: v = x_1 - x_0
        target_velocity = x_1 - x_0
        
        # Model prediction
        # Note: Wan2.2 model expects timestep in [0, 1000] range typically
        # For flow matching, we scale t from [0, 1] to model's expected range
        timestep_scaled = t * 1000.0
        
        predicted_velocity = self.model(
            x=x_t,
            timestep=timestep_scaled,
            context=context,
        )
        
        # MSE loss between predicted and target velocity
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='mean')
        
        # Additional metrics
        with torch.no_grad():
            # Velocity magnitude
            pred_mag = predicted_velocity.abs().mean()
            target_mag = target_velocity.abs().mean()
            
            # Cosine similarity (direction alignment)
            pred_flat = predicted_velocity.flatten(1)
            target_flat = target_velocity.flatten(1)
            cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        
        return {
            'loss': loss,
            'pred_velocity_mag': pred_mag,
            'target_velocity_mag': target_mag,
            'velocity_cos_sim': cos_sim,
        }
    
    def train_step(
        self,
        videos: torch.Tensor,
        texts: list,
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            videos: Batch of videos [B, C, T, H, W] or pre-encoded latents
            texts: List of text prompts
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move to device and dtype
        
        videos = videos.to(self.device, dtype=self.dtype)
        
        # Encode to latent space if needed
        if self.vae is not None and videos.shape[1] != 48:
            x_1 = self.encode_video(videos)
        else:
            x_1 = videos
        
        context = texts.to(self.device, dtype=self.dtype)
        
        # Sample timesteps
        batch_size = x_1.shape[0]
        t = self.sample_timesteps(batch_size)
        
        # Compute loss
        loss_dict = self.compute_flow_matching_loss(x_1, t, context)
        loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.grad_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update EMA
        if self.ema_model is not None:
            self.update_ema()
        
        # Increment step counter
        self.global_step += 1
        
        # Gather metrics
        metrics = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'pred_velocity_mag': loss_dict['pred_velocity_mag'].item(),
            'target_velocity_mag': loss_dict['target_velocity_mag'].item(),
            'velocity_cos_sim': loss_dict['velocity_cos_sim'].item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        return metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader yielding (videos, texts) pairs
            scheduler: Optional learning rate scheduler
            
        Returns:
            Average metrics for the epoch
        """
        self.epoch += 1
        epoch_metrics = {}
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {self.epoch}",
            dynamic_ncols=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch (handle different formats)
            if isinstance(batch, (list, tuple)):
                videos, texts = batch[0], batch[1]
            else:
                videos = batch['video']
                texts = batch['text']
            
            # Training step
            metrics = self.train_step(videos, texts)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'grad': f"{metrics['grad_norm']:.2f}",
                'cos_sim': f"{metrics['velocity_cos_sim']:.3f}",
            })
            
            # Step scheduler if per-step scheduling
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                scheduler.step_update(self.global_step)
        
        # Average metrics
        avg_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        # Step scheduler if per-epoch scheduling
        if scheduler is not None and not hasattr(scheduler, 'step_update'):
            scheduler.step()
        
        logger.info(f"Epoch {self.epoch} - Loss: {avg_metrics['loss']:.4f}, "
                   f"Cos Sim: {avg_metrics['velocity_cos_sim']:.3f}")
        
        return avg_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        safe_torch.save_file(checkpoint, path)
        # torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.ema_model is not None and checkpoint['ema_model_state_dict'] is not None:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Resumed at epoch {self.epoch}, step {self.global_step}")

EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.png', '.jpg', '.jpeg', '.webp', '.webm']

class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str = "./dataset/", vae: Optional[WanVAE] = None):
        super().__init__()
        if vae is None:
            raise ValueError("VAE must be provided for encoding videos.")
        self.dataset_path = Path(dataset_path)
        self.img_paths = [
            p for p in self.dataset_path.rglob('*') 
            if p.suffix.lower() in EXTENSIONS
        ]
        self.precalculated_latents = {}
        for path in self.img_paths:
            # b, c, f, h, w
            # f is frames, leave as 1 for images
            image = Image.open(path).convert('RGB')
            image = np.array(image).astype(np.float32) / 127.5 - 1.0
            #rescale so res is mod 8
            h, w, _ = image.shape
            new_h = (h // 16) * 16
            new_w = (w // 16) * 16
            image = image[:new_h, :new_w, :]
            image_tensor = torch.from_numpy(image)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            #frames
            image_tensor = image_tensor.unsqueeze(2)  # [1, C, 1, H, W]
            image_tensor = image_tensor.to("cuda")
            latent = vae.encode(image_tensor)
            self.precalculated_latents[str(path)] = latent.squeeze(0)
        
        self.combinations = []
        for img_path in self.img_paths:
            emb_path = img_path.with_suffix('.safetensors')
            if emb_path.exists():
                self.combinations.append((self.precalculated_latents.get(str(img_path)), emb_path))
            
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx):
        latent, emb_path = self.combinations[idx]
        embedding = safe_torch.load_file(emb_path)['embeddings']
        return (latent, embedding)
        
        
def main():
    """
    Example training script
    """
    # Initialize VAE for encoding in dataset
    vae = WanVAE(
        z_dim=48, dim_mult = [1, 2, 4, 4], num_res_blocks = 2, attn_scales = [], temperal_downsample = [False, True, True], dropout = 0.0
    )
    vae.load_state_dict(safe_torch.load_file("models/vae/vae2.2.safetensors"))
    vae.to("cuda").eval()
    dataset = FolderDataset(dataset_path="./dataset/", vae=vae)
    # deload vae since not useful
    del vae
    torch.cuda.empty_cache()
    print('setup done')
    # Initialize model components
    # Load checkpoint to check number of layers    
    # Count the number of blocks in the checkpoint
    num_blocks_in_checkpoint = 30
    model = WanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        in_dim=48,  # Wan2.2 uses 48-channel latents
        out_dim=48,
        dim=3072,
        ffn_dim=8192,
        num_heads=16,
        num_layers=num_blocks_in_checkpoint,  # Use the actual number from checkpoint
        text_len=512,
        text_dim=4096,
    )
    safe_torch.load_model(model, "models/wan2.2.safetensors", device="cuda")
    model.to("cuda").train()
    print('model loaded')
    # Initialize trainer
    trainer = FlowMatchingTrainer(
        model=model,
        vae=vae,
        learning_rate=1e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        ema_decay=None,
        device="cuda",
        dtype=torch.bfloat16,
    )
    print('trainer initialized')
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=5,  # Number of epochs
        eta_min=1e-6,
    )
    print('dataloader and scheduler set up')
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, scheduler)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {metrics['loss']:.4f}, Cos Sim: {metrics['velocity_cos_sim']:.3f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/wan22_flow_epoch_{epoch+1}.pt")
    
    # Save final model
    trainer.save_checkpoint("checkpoints/wan22_flow_final.pt")
    logger.info("Training complete!")


if __name__ == "__main__":
    
    main()
