import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import safetensors.torch as safe_torch

from comfywan import WanVAE

EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.png', '.jpg', '.jpeg', '.webp', '.webm']
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

logger = logging.getLogger(__name__)


def _load_vae(checkpoint_path: Path, device: torch.device) -> WanVAE:
    vae = WanVAE(
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    state_dict = safe_torch.load_file(str(checkpoint_path))
    vae.load_state_dict(state_dict)
    vae.to(device).eval()
    return vae


def _image_to_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    array = np.array(image, dtype=np.float32) / 127.5 - 1.0
    height, width, _ = array.shape
    height = (height // 16) * 16
    width = (width // 16) * 16
    if height == 0 or width == 0:
        raise ValueError(f"Image {image_path} is too small after cropping to multiples of 16")
    array = array[:height, :width, :]
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.unsqueeze(2)  # add temporal dim
    return tensor


def precalculate_latents(
    dataset_path: Path,
    vae: WanVAE,
    device: torch.device,
    overwrite: bool = False,
    decode_test: bool = False,
):
    image_paths = [p for p in dataset_path.rglob('*') if p.suffix.lower() in EXTENSIONS]
    if not image_paths:
        logger.warning("No images or videos found for latent precalculation")
        return

    vae = vae.to(device)
    logger.info("Found %d media files to process", len(image_paths))
    for path in image_paths:
        latent_path = path.with_suffix('.latent.safetensors')
        if latent_path.exists() and not overwrite:
            logger.info("Skipping %s, latent already exists", path)
            continue

        try:
            tensor = _image_to_tensor(path).to(device)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path, exc)
            continue
        with torch.no_grad():
            latent = vae.encode(tensor)
        latent_cpu = latent.squeeze(0).cpu()
        safe_torch.save_file({'latent': latent_cpu}, str(latent_path))
        logger.info("Saved latent for %s", path)

        if decode_test and path.suffix.lower() in IMAGE_EXTENSIONS:
            decoded = vae.decode(latent)
            decoded = decoded.squeeze(0).squeeze(1)
            decoded = torch.clamp(decoded, -1.0, 1.0)
            decoded_img = ((decoded.permute(1, 2, 0) + 1.0) * 127.5)
            decoded_img = decoded_img.detach().numpy().astype(np.uint8)
            decoded_path = path.with_name(f"{path.stem}.decoded{path.suffix}")
            Image.fromarray(decoded_img).save(decoded_path)
            logger.info("Saved decode test image to %s", decoded_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for dataset images")
    parser.add_argument("--dataset-path", default="./dataset", type=Path, help="Directory containing image files")
    parser.add_argument(
        "--vae-checkpoint",
        default=Path("models/vae2.2.safetensors"),
        type=Path,
        help="Path to Wan VAE checkpoint",
    )
    parser.add_argument("--device", default="cuda", help="Device to run VAE encoding on")
    parser.add_argument("--overwrite", action="store_true", help="Recompute latents even if they exist")
    parser.add_argument("--decode-test", action="store_true", help="Decode latents and save reconstructions for inspection")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    dataset_path = args.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

    vae = _load_vae(args.vae_checkpoint, device)
    precalculate_latents(dataset_path, vae, device, overwrite=args.overwrite, decode_test=args.decode_test)


if __name__ == "__main__":
    main()
