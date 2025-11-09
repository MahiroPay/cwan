"""Simple inference script for Wan2.2 flow-matching checkpoints.

This utility samples latents by iterating the rectified-flow update that Wan22
uses in ComfyUI.  It expects precomputed text embeddings (conditioned and
unconditioned) saved as Safetensors as produced by ``precalculate_text_embeds``.
Optionally, the script can decode the final latent with the Wan VAE and save a
PNG preview of the first frame.

Memory Optimizations:
- Use --offload-to-cpu to enable block offloading for memory-efficient inference
- Reduces GPU memory requirements but slows inference by ~2-3x
- See MEMORY_OPTIMIZATIONS.md for detailed documentation
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional
import torch.profiler
import safetensors.torch as safe_torch
import torch
from PIL import Image

from comfywan import WanModel, WanVAE, Wan22LatentFormat


def load_embeddings(path: pathlib.Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Load a text embedding tensor from a Safetensors file."""
    data = safe_torch.load_file(str(path))
    if "embeddings" not in data:
        raise KeyError(f"Embeddings tensor not found in {path}")
    tensor = data["embeddings"].to(device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> WanModel:
    """Instantiate WanModel and load weights."""
    model = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        in_dim=48,
        out_dim=48,
        dim=3072,
        ffn_dim=14336,
        num_heads=16,
        num_layers=args.num_layers,
        text_len=512,
        text_dim=4096,
        offload_to_cpu=args.offload_to_cpu,
    )
    state_dict = safe_torch.load_file(str(args.checkpoint), device="cpu")
    model.load_state_dict(state_dict)
    
    if args.offload_to_cpu:
        print("Block offloading to CPU enabled for memory-efficient inference")
    
    return model.to(device=device, dtype=dtype).eval()


def build_vae(args: argparse.Namespace, device: torch.device) -> Optional[WanVAE]:
    if args.vae_checkpoint is None:
        return None
    vae = WanVAE(
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    vae_sd = safe_torch.load_file(str(args.vae_checkpoint), device="cpu")
    vae.load_state_dict(vae_sd)
    return vae.to(device=device).eval()


def predict_x0(
    model: WanModel,
    latent: torch.Tensor,
    sigma_value: float,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    cfg_scale: float,
    sigma_multiplier: float,
) -> torch.Tensor:
    """Return CFG-combined x0 prediction for a single sigma."""
    with torch.no_grad():  # ✅ Add this!
        batch = latent.shape[0]
        sigma_tensor = torch.full((batch,), sigma_value, device=latent.device, dtype=torch.float32)
        timestep = sigma_tensor * sigma_multiplier

        cond_in = cond.expand(batch, -1, -1).to(latent.dtype)
        uncond_in = uncond.expand(batch, -1, -1).to(latent.dtype)

        pred_cond = model(x=latent, timestep=timestep, context=cond_in)
        pred_uncond = model(x=latent, timestep=timestep, context=uncond_in)

        residual = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        sigma_broadcast = sigma_tensor.view(batch, 1, 1, 1, 1).to(latent.dtype)
        x0 = latent - sigma_broadcast * residual
        return x0


def rectified_flow_sample(
    model: WanModel,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    latent_shape: tuple[int, ...],
    steps: int,
    cfg_scale: float,
    sigma_start: float,
    sigma_end: float,
    sigma_multiplier: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():  # ✅ Add outer context too!
        torch.manual_seed(seed)
        latent = torch.randn(latent_shape, device=device, dtype=dtype)
        sigmas = torch.linspace(sigma_start, sigma_end, steps + 1, device=device, dtype=torch.float32)
        
        for i in range(steps):
            print(i)
            sigma_curr = float(sigmas[i].item())
            sigma_next = float(sigmas[i + 1].item())
            if sigma_curr <= 0.0:
                break
            x0 = predict_x0(model, latent, sigma_curr, cond, uncond, cfg_scale, sigma_multiplier)
            if sigma_next <= 0.0:
                latent = x0
                break
            ratio = sigma_next / sigma_curr  # ✅ Use Python float directly
            latent = ratio * latent + (1.0 - ratio) * x0
            
            # ✅ Optional: Force garbage collection every N steps
            if i % 5 == 0:
                torch.cuda.empty_cache()
        
        return latent


def save_preview(
    vae: WanVAE,
    latent_format: Wan22LatentFormat,
    latent: torch.Tensor,
    output_path: pathlib.Path,
) -> None:
    """Decode the first sample/frame and save as PNG."""
    with torch.no_grad():
        latent_raw = latent_format.process_out(latent.float())
        decoded = vae.decode(latent_raw)
    decoded = decoded[0, :, 0]  # [C, H, W]
    decoded = torch.clamp(decoded, -1.0, 1.0)
    decoded = (decoded + 1.0) * 127.5
    array = decoded.permute(1, 2, 0).cpu().numpy().astype("uint8")
    Image.fromarray(array).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2 rectified-flow inference test")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to Wan2.2 flow checkpoint")
    parser.add_argument("--cond", type=pathlib.Path, required=True, help="Safetensors file with conditioned embeddings")
    parser.add_argument("--uncond", type=pathlib.Path, required=True, help="Safetensors file with unconditional embeddings")
    parser.add_argument("--vae-checkpoint", type=pathlib.Path, default=None, help="Optional Wan VAE checkpoint for decoding")
    parser.add_argument("--output-latent", type=pathlib.Path, default=pathlib.Path("wan22_sample.latent.safetensors"))
    parser.add_argument("--preview", type=pathlib.Path, default=None, help="Optional PNG path for decoded preview")
    parser.add_argument("--width", type=int, default=512, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Output height in pixels")
    parser.add_argument("--latent-width", type=int, default=None, help="Optional latent width override")
    parser.add_argument("--latent-height", type=int, default=None, help="Optional latent height override")
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to sample")
    parser.add_argument("--steps", type=int, default=30, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--sigma-start", type=float, default=0.999, help="Starting sigma (close to 1)")
    parser.add_argument("--sigma-end", type=float, default=0.01, help="Ending sigma (close to 0)")
    parser.add_argument("--sigma-shift", type=float, default=8.0, help="Flow shift used during training")
    parser.add_argument("--sigma-multiplier", type=float, default=1000.0, help="Multiplier to convert sigma to model timestep")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=30, help="Wan transformer layers")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable block offloading to CPU for memory-efficient inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    latent_format = Wan22LatentFormat()
    model = build_model(args, device, dtype)

    cond = load_embeddings(args.cond, device, dtype)
    uncond = load_embeddings(args.uncond, device, dtype)

    default_factor = 16
    h_lat = args.latent_height if args.latent_height is not None else max(1, args.height // default_factor)
    w_lat = args.latent_width if args.latent_width is not None else max(1, args.width // default_factor)
    if args.latent_height is None and args.height % default_factor != 0:
        raise ValueError("Height must be divisible by 16 or provide --latent-height")
    if args.latent_width is None and args.width % default_factor != 0:
        raise ValueError("Width must be divisible by 16 or provide --latent-width")
    latent_shape = (
        args.batch_size,
        latent_format.latent_channels,
        args.frames,
        h_lat,
        w_lat,
    )

    sampled_latent = rectified_flow_sample(
        model=model,
        cond=cond,
        uncond=uncond,
        latent_shape=latent_shape,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        sigma_multiplier=args.sigma_multiplier,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    safe_torch.save_file({"latent": sampled_latent.cpu()}, str(args.output_latent))

    if args.preview is not None:
        vae = build_vae(args, device)
        if vae is None:
            raise ValueError("--preview requested but no --vae-checkpoint provided")
        preview_path = args.preview
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        save_preview(vae, latent_format, sampled_latent, preview_path)
        print(f"Saved preview to {preview_path}")

    print(f"Saved latent sample to {args.output_latent}")


if __name__ == "__main__":
    main()
