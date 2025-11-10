import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def _image_to_tensor(image_path: Path, max_megapixels: float = None) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    
    # Rescale if image exceeds max megapixels
    if max_megapixels is not None:
        width, height = image.size
        current_mp = (width * height) / 1_000_000
        if current_mp > max_megapixels:
            scale_factor = (max_megapixels / current_mp) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Rescaled {image_path.name} from {width}x{height} ({current_mp:.2f}MP) to {new_width}x{new_height} ({max_megapixels:.2f}MP)")
    
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


def _encode_tiled(vae: WanVAE, tensor: torch.Tensor, tile_size: int = 512, overlap: int = 64) -> torch.Tensor:
    """
    Encode a tensor using tiled VAE encoding to reduce memory usage.
    
    Args:
        vae: The VAE model
        tensor: Input tensor of shape [B, C, T, H, W]
        tile_size: Size of each tile (must be multiple of 16)
        overlap: Overlap between tiles to reduce seam artifacts
    
    Returns:
        Encoded latent tensor
    """
    b, c, t, h, w = tensor.shape
    
    # Ensure tile_size and overlap are multiples of 16
    tile_size = (tile_size // 16) * 16
    overlap = (overlap // 16) * 16
    
    # If image is smaller than tile_size, just encode normally
    if h <= tile_size and w <= tile_size:
        return vae.encode(tensor)
    
    # Calculate stride (tile_size - overlap)
    stride = tile_size - overlap
    
    # Calculate number of tiles needed
    n_tiles_h = (h - overlap + stride - 1) // stride
    n_tiles_w = (w - overlap + stride - 1) // stride
    
    # Calculate output dimensions (VAE has 8x spatial downsampling)
    latent_h = h // 8
    latent_w = w // 8
    
    # Get latent channels by encoding a small sample
    with torch.no_grad():
        sample_size = min(tile_size, h, w)
        sample = tensor[:, :, :, :sample_size, :sample_size]
        sample_latent = vae.encode(sample)
        latent_c = sample_latent.shape[1]
        del sample, sample_latent
    
    # Initialize output latent and weight map for blending
    latent = torch.zeros(b, latent_c, t, latent_h, latent_w, device=tensor.device, dtype=tensor.dtype)
    weights = torch.zeros(b, 1, t, latent_h, latent_w, device=tensor.device, dtype=tensor.dtype)
    
    logger.info(f"Encoding with tiles: {n_tiles_h}x{n_tiles_w} tiles of size {tile_size}x{tile_size} with {overlap}px overlap")
    
    # Process each tile
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile boundaries in input space
            h_start = i * stride
            h_end = min(h_start + tile_size, h)
            w_start = j * stride
            w_end = min(w_start + tile_size, w)
            
            # Extract tile
            tile = tensor[:, :, :, h_start:h_end, w_start:w_end]
            
            # Encode tile
            with torch.no_grad():
                tile_latent = vae.encode(tile)
            
            # Get actual encoded tile dimensions
            tile_h, tile_w = tile_latent.shape[3], tile_latent.shape[4]
            
            # Calculate tile boundaries in latent space based on actual encoded size
            lh_start = h_start // 8
            lw_start = w_start // 8
            lh_end = lh_start + tile_h
            lw_end = lw_start + tile_w
            
            # Create weight mask for blending (feather edges) matching actual tile size
            weight_mask = torch.ones(1, 1, t, tile_h, tile_w, device=tensor.device, dtype=tensor.dtype)
            
            # Apply feathering at overlaps
            feather_size = overlap // 8  # Convert to latent space
            if feather_size > 0:
                # Calculate actual feather size based on tile dimensions
                feather_w = min(feather_size, tile_w)
                feather_h = min(feather_size, tile_h)
                
                # Feather left edge
                if w_start > 0 and feather_w > 0:
                    for k in range(min(feather_w, tile_w)):
                        weight_mask[:, :, :, :, k] *= (k + 1) / (feather_w + 1)
                # Feather right edge
                if w_end < w and feather_w > 0:
                    for k in range(min(feather_w, tile_w)):
                        if k < tile_w:
                            weight_mask[:, :, :, :, -(k + 1)] *= (k + 1) / (feather_w + 1)
                # Feather top edge
                if h_start > 0 and feather_h > 0:
                    for k in range(min(feather_h, tile_h)):
                        weight_mask[:, :, :, k, :] *= (k + 1) / (feather_h + 1)
                # Feather bottom edge
                if h_end < h and feather_h > 0:
                    for k in range(min(feather_h, tile_h)):
                        if k < tile_h:
                            weight_mask[:, :, :, -(k + 1), :] *= (k + 1) / (feather_h + 1)
            
            # Accumulate weighted tile
            latent[:, :, :, lh_start:lh_end, lw_start:lw_end] += tile_latent * weight_mask
            weights[:, :, :, lh_start:lh_end, lw_start:lw_end] += weight_mask
    
    # Normalize by accumulated weights
    latent = latent / weights.clamp(min=1e-8)
    
    return latent


def precalculate_latents(
    dataset_path: Path,
    vae: WanVAE,
    device: torch.device,
    overwrite: bool = False,
    decode_test: bool = False,
    tile_size: int = 512,
    tile_overlap: int = 64,
    max_megapixels: float = None,
):
    image_paths = [p for p in dataset_path.rglob('*') if p.suffix.lower() in EXTENSIONS  and ".ipynb_checkpoints" not in str(p)]
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
            tensor = _image_to_tensor(path, max_megapixels=max_megapixels).to(device)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path, exc)
            continue
        
        # Try normal encoding first, fall back to tiled encoding on OOM
        try:
            with torch.no_grad():
                latent = vae.encode(tensor)
            logger.info("Encoded %s using standard method", path)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning("OOM during encoding %s, falling back to tiled encoding", path)
                torch.cuda.empty_cache()  # Clear cache before retrying
                try:
                    latent = _encode_tiled(vae, tensor, tile_size=tile_size, overlap=tile_overlap)
                    logger.info("Successfully encoded %s using tiled method", path)
                except Exception as tiled_exc:
                    logger.error("Tiled encoding also failed for %s: %s", path, tiled_exc)
                    continue
            else:
                logger.error("Failed to encode %s: %s", path, e)
                continue
        
        latent_cpu = latent.squeeze(0).cpu()
        safe_torch.save_file({'latent': latent_cpu}, str(latent_path))
        logger.info("Saved latent for %s", path)

        if decode_test and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                decoded = vae.decode(latent)
                decoded = decoded.squeeze(0).squeeze(1)
                decoded = torch.clamp(decoded, -1.0, 1.0)
                decoded_img = ((decoded.permute(1, 2, 0) + 1.0) * 127.5)
                decoded_img = decoded_img.detach().cpu().numpy().astype(np.uint8)
                decoded_path = path.with_name(f"{path.stem}.decoded{path.suffix}")
                Image.fromarray(decoded_img).save(decoded_path)
                logger.info("Saved decode test image to %s", decoded_path)
            except Exception as decode_exc:
                logger.warning("Failed to decode test for %s: %s", path, decode_exc)


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
    parser.add_argument("--tile-size", default=512, type=int, help="Tile size for tiled encoding (must be multiple of 16)")
    parser.add_argument("--tile-overlap", default=64, type=int, help="Overlap between tiles for blending (must be multiple of 16)")
    parser.add_argument("--max-megapixels", type=float, default=None, help="Maximum megapixels for images (rescale if exceeded)")
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
    precalculate_latents(
        dataset_path, 
        vae, 
        device, 
        overwrite=args.overwrite, 
        decode_test=args.decode_test,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        max_megapixels=args.max_megapixels,
    )


if __name__ == "__main__":
    main()
