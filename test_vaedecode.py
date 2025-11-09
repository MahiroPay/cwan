
import argparse
import pathlib
from typing import Optional
import torch.profiler
import safetensors.torch as safe_torch
import torch
from PIL import Image

from comfywan import WanModel, WanVAE, Wan22LatentFormat


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
    parser.add_argument("--vae-checkpoint", type=pathlib.Path, default=None, help="Wan VAE checkpoint for decoding")
    parser.add_argument("--input-latent", type=pathlib.Path, default=pathlib.Path("wan22_sample.latent.safetensors"))
    parser.add_argument("--save-to", type=pathlib.Path, default=None, help="Optional PNG path for decoded preview")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    latent_format = Wan22LatentFormat()
    vae = build_vae(args, device)
    sampled_latent = safe_torch.load_file(args.input_latent, device=device)['latent']
    preview_path = args.save_to
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview(vae, latent_format, sampled_latent, preview_path)
    print(f"Saved preview to {preview_path}")