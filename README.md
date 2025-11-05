# comfywan cursed implementation

90% vibecoded, 10% comfycoded and 0% type checking

## Dataset Format
```
dataset/
 - {name}.{extension}
 - {name}.txt
```
valid extensions: `.mp4, .avi, .mov, .mkv, .png, .jpg, .jpeg, .webp, .webm`

## Multi-GPU Training

The training script now supports multi-GPU training on a single node using PyTorch's DistributedDataParallel.

### Quick Start

**Single GPU:**
```bash
python train_flow_matching.py --batch_size 1 --num_epochs 100
```

**Multi-GPU (2 GPUs):**
```bash
./train_multigpu.sh 2 --batch_size 2 --num_epochs 50
```

For detailed instructions and examples, see [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md)