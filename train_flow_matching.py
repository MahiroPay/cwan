"""
Flow Matching Training Loop for Wan2.2

This implements a flow matching training approach for the Wan2.2 video diffusion model.
Flow matching learns a vector field that transports noise to data through optimal transport paths.

Key concepts:
- Flow matching uses conditional probability paths from noise (t=0) to data (t=1)
- The model predicts the residual r = x_0 - x_1 that recovers the clean latent via x_t - sigma * r
- Training objective: E[||x_t - sigma * f_theta(x_t, sigma, c) - x_1||^2]
- Rectified-flow interpolation: x_sigma = sigma * x_0 + (1 - sigma) * x_1 where x_0~N(0,I), x_1~data

Memory Optimizations:
- Use --gradient_checkpointing to reduce memory usage by ~50-70% (slower by ~30-40%)
- See MEMORY_OPTIMIZATIONS.md for detailed documentation

Reference:
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Stable Video Diffusion" uses rectified flow
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm
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
    - x_sigma = sigma * x_0 + (1 - sigma) * x_1 for sigma in (0, 1)
    - Model predicts residual r = x_0 - x_1 so that x_1 = x_sigma - sigma * r
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
        sigma_shift: float = 8.0,
        sigma_multiplier: float = 1000.0,
        sigma_eps: float = 1e-4,
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
            sigma_shift: Flow schedule shift, matches ComfyUI's ModelSamplingDiscreteFlow
            sigma_multiplier: Scaling used when converting sigma to timesteps for Wan22
            sigma_eps: Small constant to avoid degenerate sigma values at the bounds
        """
        self.device = device
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.sigma_shift = sigma_shift
        self.sigma_multiplier = sigma_multiplier
        self.sigma_eps = sigma_eps
        
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
    
    def sample_sigmas(self, batch_size: int) -> torch.Tensor:
        """
        Sample noise levels (sigma) following ComfyUI's rectified-flow schedule.

        Args:
            batch_size: Number of sigma values to sample.

        Returns:
            Tensor of shape [batch_size] with sigma in (0, 1).
        """
        u = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        u = u.clamp_(self.sigma_eps, 1.0 - self.sigma_eps)
        sigma = self.sigma_shift * u / (1.0 + (self.sigma_shift - 1.0) * u)
        return sigma
    
    def get_noisy_latent(
        self,
        x_1: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create noisy latent via linear interpolation
        
        Rectified-flow interpolant used by ComfyUI:
        x_sigma = sigma * x_0 + (1 - sigma) * x_1
        where x_0 ~ N(0, I) is pure noise and x_1 is the data latent.
        
        Args:
            x_1: Clean data latent [B, C, T, H, W]
            sigma: Noise levels [B] in (0, 1)
            
        Returns:
            x_t: Interpolated latent [B, C, T, H, W]
            x_0: Source noise [B, C, T, H, W]
        """
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # Reshape sigma for broadcasting: [B] -> [B, 1, 1, 1, 1]
        sigma_broadcast = sigma.view(-1, 1, 1, 1, 1).to(x_1.dtype)
        
        # Rectified-flow interpolation consistent with ComfyUI
        x_t = sigma_broadcast * x_0 + (1 - sigma_broadcast) * x_1
        
        return x_t, x_0
    
    def compute_flow_matching_loss(
        self,
        x_1: torch.Tensor,
        sigma: torch.Tensor,
        context: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
        x_0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the rectified-flow loss aligned with ComfyUI's implementation.

        The Wan2.2 flow backbone is trained so that
            x_0 = x_sigma - sigma * f_theta(x_sigma, sigma, c)
        matches the clean latent. This is equivalent to predicting the
        residual (noise - data) under the rectified-flow interpolant.

        Args:
            x_1: Clean data latent [B, C, T, H, W]
            sigma: Noise levels [B] in (0, 1)
            context: Text conditioning [B, L, D]
            x_t: Optional precomputed noisy latent
            x_0: Optional precomputed noise sample

        Returns:
            Dictionary with loss scalar and diagnostic metrics.
        """
        sigma = sigma.to(self.device, dtype=torch.float32)

        if x_t is None or x_0 is None:
            x_t, x_0 = self.get_noisy_latent(x_1, sigma)

        sigma_broadcast = sigma.view(-1, 1, 1, 1, 1).to(x_t.dtype)
        timestep_scaled = sigma * self.sigma_multiplier

        predicted_residual = self.model(
            x=x_t,
            timestep=timestep_scaled,
            context=context,
        )
        predicted_residual = predicted_residual.to(x_t.dtype)

        x0_pred = x_t - sigma_broadcast * predicted_residual
        loss = F.mse_loss(x0_pred.float(), x_1.float(), reduction='mean')

        with torch.no_grad():
            target_residual = (x_0 - x_1).float()
            pred_flat = predicted_residual.float().flatten(1)
            target_flat = target_residual.flatten(1)
            cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
            residual_mse = F.mse_loss(pred_flat, target_flat, reduction='mean')
            pred_mag = pred_flat.abs().mean()
            target_mag = target_flat.abs().mean()
            sigma_mean = sigma.mean()

        return {
            'loss': loss,
            'residual_mse': residual_mse,
            'pred_residual_mag': pred_mag,
            'target_residual_mag': target_mag,
            'residual_cos_sim': cos_sim,
            'sigma_mean': sigma_mean,
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
            texts: Batched text embeddings [B, L, D] or list of per-sample tensors
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move to device and dtype
        videos = videos.to(self.device)

        # Encode to latent space if needed and normalize latents
        if self.vae is not None and videos.shape[1] != self.latent_format.latent_channels:
            x_1 = self.encode_video(videos.to(dtype=torch.float32))
        else:
            x_1 = self.latent_format.process_in(videos.to(dtype=self.dtype))

        x_1 = x_1.to(self.device, dtype=self.dtype)

        if isinstance(texts, (list, tuple)):
            context = torch.stack(texts, dim=0)
        else:
            context = texts
        context = context.to(self.device, dtype=self.dtype)

        # Sample noise levels
        batch_size = x_1.shape[0]
        sigma = self.sample_sigmas(batch_size)
        
        # Compute loss
        loss_dict = self.compute_flow_matching_loss(x_1, sigma, context)
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
            'residual_mse': loss_dict['residual_mse'].item(),
            'pred_residual_mag': loss_dict['pred_residual_mag'].item(),
            'target_residual_mag': loss_dict['target_residual_mag'].item(),
            'residual_cos_sim': loss_dict['residual_cos_sim'].item(),
            'sigma_mean': loss_dict['sigma_mean'].item(),
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
                'cos_sim': f"{metrics['residual_cos_sim']:.3f}",
                'sigma': f"{metrics['sigma_mean']:.3f}",
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
        
        logger.info(
            f"Epoch {self.epoch} - Loss: {avg_metrics['loss']:.4f}, "
            f"Cos Sim: {avg_metrics['residual_cos_sim']:.3f}"
        )
        
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
    def __init__(self, dataset_path: str = "./dataset/"):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist")

        self.combinations = []
        for img_path in self.dataset_path.rglob('*'):
            if img_path.suffix.lower() not in EXTENSIONS:
                continue
            emb_path = img_path.with_suffix('.safetensors')
            latent_path = img_path.with_suffix('.latent.safetensors')
            if emb_path.exists() and latent_path.exists():
                self.combinations.append((latent_path, emb_path))

        if not self.combinations:
            raise ValueError(
                "No latent/embedding pairs found. Run precalculate_image_embeds.py before training."
            )
        
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx):
        latent_path, emb_path = self.combinations[idx]
        latent = safe_torch.load_file(str(latent_path))['latent'].float()
        embedding = safe_torch.load_file(str(emb_path))['embeddings'].float()
        return latent, embedding
        
import argparse 
def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Wan2.2 with Flow Matching')
    parser.add_argument('--dataset_path', type=str, default='./dataset/',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (usually auto-set)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing for memory-efficient training')
    parser.add_argument('--vae_gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing for VAE (if used)')
    return parser.parse_args()


def main():
    """
    Training script with multi-GPU support
    """
    # Parse arguments
    args = parse_args()
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    dataset = FolderDataset(dataset_path=args.dataset_path)
    
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
        ffn_dim=14336,
        num_heads=16,
        num_layers=num_blocks_in_checkpoint,  # Use the actual number from checkpoint
        text_len=512,
        text_dim=4096,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Load pretrained weights on CPU first, then move to device
    mod = safe_torch.load_file("models/wan2.2_ti2v_5B_fp16.safetensors", device="cpu")
    model.load_state_dict(mod)
    
    # Move model to device and set dtype
    model = model.to(device).bfloat16()
    
    # Log memory optimization settings
    if args.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled for main model")
    
    
    # Initialize trainer with distributed settings
    trainer = FlowMatchingTrainer(
        model=model,
        vae=None,
        learning_rate=1e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        ema_decay=None,
        device="cuda",
        dtype=torch.bfloat16
    )
    
    sampler = None
    shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )
    print('dataloader and scheduler set up')
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, scheduler)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {metrics['loss']:.4f}, "
            f"Cos Sim: {metrics['residual_cos_sim']:.3f}"
        )
        
        # Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/wan22_flow_epoch_{epoch+1}.pt")
    
    # Save final model
    trainer.save_checkpoint("checkpoints/wan22_flow_final.pt")
    logger.info("Training complete!")


if __name__ == "__main__":
    
    main()
