"""Alternative Flow Matching training entry point that bootstraps the Wan 2.2
model through ComfyUI's loader stack. This allows us to keep parity with the
runtime configuration ComfyUI expects (custom ops, latent formats, etc.) while
re-using the flow-matching trainer defined in ``train_flow_matching.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch

import comfy.sd

from train_flow_matching import FlowMatchingTrainer, FolderDataset

logger = logging.getLogger(__name__)

DTYPE_CHOICES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Wan2.2 with Flow Matching using ComfyUI's WAN loader"
    )
    parser.add_argument("--dataset_path", type=str, default="./dataset/",
                        help="Path to latent/embedding dataset")
    parser.add_argument("--ckpt_path", type=str,
                        default="models/wan2.2_ti2v_5B_fp16.safetensors",
                        help="Wan2.2 checkpoint to load via comfy.sd.load_diffusion_model")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory used to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per optimizer step")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs to run")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay. Leave unset to disable EMA tracking")
    parser.add_argument("--use_ema", action="store_true",
                        help="Enable EMA even if decay is unset; useful for quick experiments")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device string (defaults to cuda if available)")
    parser.add_argument("--model_dtype", type=str, choices=DTYPE_CHOICES.keys(),
                        default="bf16", help="Model compute dtype")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Attempt to enable gradient checkpointing on the WAN backbone")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use torchao's 8-bit Adam optimizer")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Wrap the WAN model with torch.compile (PyTorch 2.0+)")
    parser.add_argument("--fp8_optimizations", action="store_true",
                        help="Request fp8 optimization path in comfy load options")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Reserved for distributed launchers")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_wan_model_via_comfy(
    ckpt_path: Path,
    device: str,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
    torch_compile_enabled: bool,
    fp8_optimizations: bool,
) -> Tuple[torch.nn.Module, object]:
    """Load Wan model weights through comfy.sd.load_diffusion_model."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist")

    model_options = {"dtype": dtype}
    if fp8_optimizations:
        model_options["fp8_optimizations"] = True

    logger.info("Loading WAN weights via comfy.sd from %s", ckpt_path)
    comfy_model = comfy.sd.load_diffusion_model(str(ckpt_path), model_options=model_options)
    wan_model = comfy_model.diffusion_model

    # Ensure model lives on the requested device/dtype for training.
    wan_model = wan_model.to(device=device, dtype=dtype)

    if gradient_checkpointing:
        if hasattr(wan_model, "enable_gradient_checkpointing"):
            wan_model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled on WAN backbone")
        else:
            logger.warning("WAN model does not expose gradient checkpointing hooks; flag ignored")
    elif hasattr(wan_model, "disable_gradient_checkpointing"):
        wan_model.disable_gradient_checkpointing()

    if torch_compile_enabled:
        if hasattr(torch, "compile"):
            logger.info("Compiling WAN model with torch.compile")
            wan_model = torch.compile(wan_model)
        else:
            logger.warning("torch.compile requested but not available in this PyTorch build")

    return wan_model, comfy_model.latent_format


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = resolve_device(args.device)
    dtype = DTYPE_CHOICES[args.model_dtype]

    dataset = FolderDataset(dataset_path=args.dataset_path)
    logger.info("Dataset ready with %d samples", len(dataset))

    wan_model, latent_format = load_wan_model_via_comfy(
        ckpt_path=Path(args.ckpt_path),
        device=device,
        dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile_enabled=args.torch_compile,
        fp8_optimizations=args.fp8_optimizations,
    )

    ema_decay = args.ema_decay if args.ema_decay is not None else (0.9999 if args.use_ema else None)

    trainer = FlowMatchingTrainer(
        model=wan_model,
        vae=None,
        latent_format=latent_format,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        ema_decay=ema_decay,
        device=device,
        dtype=dtype,
        use_8bit_adam=args.use_8bit_adam,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        metrics = trainer.train_epoch(dataloader, scheduler)
        logger.info(
            "Epoch %d/%d | loss %.4f | cos %.3f | grad %.2f",
            epoch + 1,
            args.num_epochs,
            metrics["loss"],
            metrics["residual_cos_sim"],
            metrics["grad_norm"],
        )

        ckpt_path = checkpoint_dir / f"wan22_comfy_flow_epoch_{epoch + 1:04d}.safetensors"
        trainer.save_checkpoint(str(ckpt_path))

    final_path = checkpoint_dir / "wan22_comfy_flow_final.safetensors"
    trainer.save_checkpoint(str(final_path))
    logger.info("Training complete; final checkpoint stored at %s", final_path)


if __name__ == "__main__":
    main()
