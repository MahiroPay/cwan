#!/bin/bash
# Multi-GPU training launcher script for Wan2.2 Flow Matching
#
# Usage:
#   Single GPU: python train_flow_matching.py [args]
#   Multi-GPU:  ./train_multigpu.sh [num_gpus] [args]
#
# Example:
#   ./train_multigpu.sh 2 --batch_size 2 --num_epochs 50

# Number of GPUs to use (default: all available)
NUM_GPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}
shift  # Remove first argument so remaining args can be passed to training script

echo "Starting training on $NUM_GPUS GPU(s)..."

# Launch training with torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_flow_matching.py "$@"
