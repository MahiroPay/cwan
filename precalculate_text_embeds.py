import argparse
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from comfy.sd import load_clip, CLIPType

def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}")
    return mapping[name]


def resolve_device(spec: str) -> torch.device:
    spec = spec.lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def gather_text_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Wan text embeddings.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Directory with .txt files.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Wan T5 checkpoint (.safetensors).")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Optional SentencePiece tokenizer path.")
    parser.add_argument("--device", type=str, default="auto", help="Device spec (auto, cpu, cuda, mps, ...).")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float16, help="Computation dtype.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if target exists.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    text_files = gather_text_files(dataset_dir)
    if not text_files:
        print("No .txt files found.")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading encoder weights from: {args.checkpoint}")
    clip = load_clip(ckpt_paths=[str(args.checkpoint)], embedding_directory=None, clip_type=CLIPType.WAN, model_options={})
    torch.set_grad_enabled(False)

    for start in range(0, len(text_files), args.batch_size):
        batch_paths = text_files[start : start + args.batch_size]
        batch_texts = []
        targets = []

        for path in batch_paths:
            target = path.with_suffix(".safetensors")
            if target.exists() and not args.overwrite:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            batch_texts.append(text)
            targets.append(target)

        if not batch_texts:
            continue

        with torch.inference_mode():
            toks = [clip.tokenize(text) for text in batch_texts]
            embeddings = [clip.encode_from_tokens_scheduled(tok) for tok in toks]
        for idx, target in enumerate(targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "embeddings": embeddings[idx][0][0].cpu().contiguous(),
                },
                str(target),
            )
            print(f"Saved {target}")

    print("Done.")


if __name__ == "__main__":
    
    main()