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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
import torch.distributed as dist
import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """
    Initialize distributed training environment.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        # Single GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        # Ensure all processes are synchronized
        dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Check if current process is the main process."""
    return rank == 0


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
        rank: int = 0,
        world_size: int = 1,
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
            rank: Process rank for distributed training
            world_size: Total number of processes for distributed training
        """
        self.device = device
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        # Model components
        self.model = model.to(device)
        
        # Wrap model with DDP for multi-GPU training
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
        
        self.vae = vae.to(device) if vae is not None else None
        self.latent_format = latent_format if latent_format is not None else Wan22LatentFormat()
        
        # Freeze VAE (only train the diffusion model)
        if self.vae is not None:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
                
        # Optimizer - get the actual model parameters (unwrap DDP if needed)
        model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()
        self.optimizer = AdamW(
            model_params,
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
        # Get the actual model (unwrap DDP if needed)
        actual_model = self.model.module if self.is_distributed else self.model
        ema_model = type(actual_model)(
            **{k: v for k, v in actual_model.__dict__.items() 
               if not k.startswith('_') and k not in ['training']}
        ).to(self.device)
        ema_model.load_state_dict(actual_model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA model parameters"""
        # Get the actual model parameters (unwrap DDP if needed)
        actual_model = self.model.module if self.is_distributed else self.model
        for ema_param, model_param in zip(
            self.ema_model.parameters(), 
            actual_model.parameters()
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
        x_0 = x_0.bfloat16()
        
        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t_broadcast = t.view(-1, 1, 1, 1, 1)
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        x_t = x_t.bfloat16()
        
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
        t = t.bfloat16()
        timestep_scaled = t * 1000.0
        context = context.bfloat16()
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
        # Get the actual model for train mode (unwrap DDP if needed)
        actual_model = self.model.module if self.is_distributed else self.model
        actual_model.train()
        
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
        model_params = actual_model.parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model_params, 
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
        
        # Only show progress bar on main process
        if is_main_process(self.rank):
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {self.epoch}",
                dynamic_ncols=True
            )
        else:
            progress_bar = dataloader
        
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
            
            # Update progress bar (only on main process)
            if is_main_process(self.rank) and hasattr(progress_bar, 'set_postfix'):
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
        
        # Log only on main process
        if is_main_process(self.rank):
            logger.info(f"Epoch {self.epoch} - Loss: {avg_metrics['loss']:.4f}, "
                       f"Cos Sim: {avg_metrics['velocity_cos_sim']:.3f}")
        
        return avg_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint (only on main process)"""
        # Only save on main process
        if not is_main_process(self.rank):
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get actual model state dict (unwrap DDP if needed)
        actual_model = self.model.module if self.is_distributed else self.model
        
        checkpoint = {
            'model_state_dict': actual_model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        # safe_torch.save_file(checkpoint['model_state_dict'], path + "statedict.safetensors")
        # safe_torch.save_file(checkpoint['optimizer_state_dict'], path + "optimizer.safetensors")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Get actual model (unwrap DDP if needed)
        actual_model = self.model.module if self.is_distributed else self.model
        actual_model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.ema_model is not None and checkpoint['ema_model_state_dict'] is not None:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        if is_main_process(self.rank):
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
    return parser.parse_args()


def main():
    """
    Training script with multi-GPU support
    """
    # Parse arguments
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device
    if world_size > 1:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Log info only on main process
    if is_main_process(rank):
        logger.info(f"Starting training with {world_size} GPU(s)")
        logger.info(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    
    # Load dataset
    dataset = FolderDataset(dataset_path=args.dataset_path)
    if is_main_process(rank):
        print(f'Dataset ready with {len(dataset)} samples')
    
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
    )
    
    # Load pretrained weights on CPU first, then move to device
    if is_main_process(rank):
        print('Loading model weights...')
    mod = safe_torch.load_file("models/wan2.2_ti2v_5B_fp16.safetensors", device="cpu")
    model.load_state_dict(mod)
    
    # Move model to device and set dtype
    model = model.to(device).bfloat16()
    
    if is_main_process(rank):
        print('Model loaded')
    
    # Initialize trainer with distributed settings
    trainer = FlowMatchingTrainer(
        model=model,
        vae=None,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        grad_clip=1.0,
        ema_decay=None,
        device=device,
        dtype=torch.bfloat16,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process(rank):
        print('Trainer initialized')
    
    # Setup dataloader with DistributedSampler for multi-GPU
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
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
    if is_main_process(rank):
        print('Dataloader and scheduler set up')
    
    # Training loop
    for epoch in range(args.num_epochs):
        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        metrics = trainer.train_epoch(dataloader, scheduler)
        
        # Log only on main process
        if is_main_process(rank):
            print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {metrics['loss']:.4f}, "
                  f"Cos Sim: {metrics['velocity_cos_sim']:.3f}")
        
        # Save checkpoint every N epochs (only on main process)
        # if (epoch + 1) % 10 == 0:
        #     trainer.save_checkpoint(f"{args.checkpoint_dir}/epoch_{epoch+1}.pt")
    
    # Save final model (only on main process)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    trainer.save_checkpoint(f"{args.checkpoint_dir}/MoeMoe.pt")
    
    if is_main_process(rank):
        logger.info("Training complete!")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()
