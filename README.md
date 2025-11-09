# comfywan cursed implementation

90% vibecoded, 10% comfycoded and 0% type checking

## Dataset Format
```
dataset/
 - {name}.{extension}
 - {name}.txt
```
valid extensions: `.mp4, .avi, .mov, .mkv, .png, .jpg, .jpeg, .webp, .webm`

## Memory Optimizations

The training and inference scripts now support memory optimizations for working with large models:

- **Gradient Checkpointing**: Reduce memory usage during training by ~50-70% at the cost of ~30-40% slower training
- **Block Offloading**: Run inference on GPUs with limited VRAM by offloading transformer blocks to CPU

### Training with Gradient Checkpointing

```bash
python train_flow_matching.py --batch_size 1 --num_epochs 100 --gradient_checkpointing
```

### Inference with Block Offloading

```bash
python test_inference.py --checkpoint models/wan2.2.safetensors --cond cond.safetensors --uncond uncond.safetensors --offload-to-cpu
```

For detailed documentation, see [MEMORY_OPTIMIZATIONS.md](MEMORY_OPTIMIZATIONS.md)

## train steps

precalc latents, text, train
```

python precalculate_text_embeds.py  --checkpoint models/umt5_xxl_fp16.safetensors --tokenizer-path models/spiece.model --dataset-dir shiina
python precalculate_image_embeds.py  --vae-checkpoint models/wan2.2_vae.safetensors --dataset-dir shiina
python train_flow_matching.py --batch_size 1 --num_epochs 15 --gradient_checkpointing --dataset-dir shiina
```