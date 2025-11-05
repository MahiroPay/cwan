# Multi-GPU Training Guide

This guide explains how to use multiple GPUs on the same node for training the Wan2.2 model with Flow Matching.

## Overview

The training script `train_flow_matching.py` now supports multi-GPU training using PyTorch's DistributedDataParallel (DDP). This allows you to:
- Train faster by distributing the workload across multiple GPUs
- Use larger batch sizes by splitting batches across GPUs
- Automatically synchronize gradients across all GPUs

## Requirements

- Multiple NVIDIA GPUs on the same machine
- PyTorch with CUDA support
- NCCL backend (usually included with PyTorch CUDA builds)

## Quick Start

### Single GPU Training (Default)

```bash
python train_flow_matching.py \
    --dataset_path ./dataset/ \
    --batch_size 1 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

### Multi-GPU Training

#### Option 1: Using the launcher script (Recommended)

```bash
# Use all available GPUs
./train_multigpu.sh --batch_size 2 --num_epochs 50

# Use specific number of GPUs (e.g., 2 GPUs)
./train_multigpu.sh 2 --batch_size 2 --num_epochs 50
```

#### Option 2: Using torch.distributed.launch directly

```bash
# For 2 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_flow_matching.py \
    --batch_size 2 \
    --num_epochs 50

# For 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_flow_matching.py \
    --batch_size 1 \
    --num_epochs 50
```

#### Option 3: Using torchrun (PyTorch >= 1.10)

```bash
# For 2 GPUs
torchrun --nproc_per_node=2 \
    train_flow_matching.py \
    --batch_size 2 \
    --num_epochs 50
```

## Command-Line Arguments

- `--dataset_path`: Path to dataset directory (default: `./dataset/`)
- `--batch_size`: Batch size **per GPU** (default: 1)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_workers`: Number of data loading workers (default: 4)
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)

## Important Notes

### Batch Size

The `--batch_size` argument specifies the batch size **per GPU**. The effective global batch size is:

```
Global Batch Size = batch_size × num_gpus
```

For example:
- 2 GPUs with `--batch_size 2` → effective batch size of 4
- 4 GPUs with `--batch_size 1` → effective batch size of 4

### Memory Considerations

- Each GPU maintains its own copy of the model
- Gradients are synchronized across GPUs during backward pass
- Adjust `--batch_size` based on your GPU memory
- Monitor GPU memory usage with `nvidia-smi`

### Checkpointing

- Checkpoints are saved only on the main process (rank 0)
- All processes can load checkpoints
- The saved checkpoint contains the unwrapped model state (without DDP wrapper)

### Data Loading

- The dataset is automatically sharded across GPUs using `DistributedSampler`
- Each GPU processes different samples in each batch
- Shuffling is handled by the sampler and synchronized across GPUs

## Performance Tips

1. **Increase batch size**: With multiple GPUs, you can use larger effective batch sizes
2. **Adjust learning rate**: Consider scaling learning rate with batch size (e.g., LR × sqrt(num_gpus))
3. **Monitor GPU utilization**: Use `nvidia-smi` or `watch -n 1 nvidia-smi` to ensure all GPUs are being used
4. **Network backend**: NCCL is automatically used for GPU communication (fastest for NVIDIA GPUs)

## Troubleshooting

### GPUs not being detected

```bash
# Check available GPUs
nvidia-smi

# Verify PyTorch can see GPUs
python -c "import torch; print(torch.cuda.device_count())"
```

### Port already in use

If you see "address already in use" errors, change the master port:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_flow_matching.py
```

### Out of memory errors

Reduce the batch size per GPU:

```bash
./train_multigpu.sh 2 --batch_size 1
```

### Process hanging at initialization

- Ensure all GPUs are accessible
- Check that NCCL is properly installed
- Verify network connectivity (for multi-node setups, though this guide focuses on single-node)

## Environment Variables

The script automatically detects distributed training settings from environment variables:
- `RANK`: Global rank of the process
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Local rank (GPU index) on the current node
- `MASTER_ADDR`: Address of the master node (automatically set by launcher)
- `MASTER_PORT`: Port for communication (automatically set by launcher)

These are automatically set by `torch.distributed.launch` or `torchrun`.

## Example Workflows

### Training on 2 GPUs with larger batch size

```bash
./train_multigpu.sh 2 \
    --batch_size 2 \
    --num_epochs 100 \
    --learning_rate 1.4e-4 \
    --checkpoint_dir ./checkpoints/2gpu_run
```

### Training on 4 GPUs with maximum throughput

```bash
./train_multigpu.sh 4 \
    --batch_size 1 \
    --num_epochs 50 \
    --num_workers 8 \
    --checkpoint_dir ./checkpoints/4gpu_run
```

## Monitoring Training

While training, you can monitor:

1. **GPU Usage** (in another terminal):
```bash
watch -n 1 nvidia-smi
```

2. **Progress**: The main process (rank 0) will show progress bars and log messages

3. **Loss curves**: All processes compute metrics, but only rank 0 logs them

## Backward Compatibility

The script remains fully compatible with single-GPU training. If you don't use distributed launch, it will automatically run on a single GPU as before:

```bash
python train_flow_matching.py  # Works as before
```
