#!/usr/bin/env python3
"""
Example usage of multi-GPU training for Wan2.2 Flow Matching.

This script demonstrates how to use the multi-GPU training functionality
with different configurations.
"""

def print_example(title, command, description=""):
    """Print a formatted example command."""
    print(f"\n{'='*70}")
    print(f"Example: {title}")
    print(f"{'='*70}")
    if description:
        print(f"{description}\n")
    print(f"Command:")
    print(f"  {command}")
    print()


def main():
    print("="*70)
    print("Wan2.2 Flow Matching - Multi-GPU Training Examples")
    print("="*70)
    
    print_example(
        "1. Single GPU Training (Default)",
        "python train_flow_matching.py --batch_size 1 --num_epochs 100",
        "Run on a single GPU with default settings."
    )
    
    print_example(
        "2. Multi-GPU Training (All Available GPUs)",
        "./train_multigpu.sh --batch_size 2 --num_epochs 50",
        "Automatically detects and uses all available GPUs."
    )
    
    print_example(
        "3. Multi-GPU Training (Specific Number of GPUs)",
        "./train_multigpu.sh 2 --batch_size 2 --num_epochs 50",
        "Use exactly 2 GPUs for training."
    )
    
    print_example(
        "4. Multi-GPU with Custom Learning Rate",
        "./train_multigpu.sh 4 --batch_size 1 --num_epochs 100 --learning_rate 2e-4",
        "Train on 4 GPUs with increased learning rate (scaled for larger batch)."
    )
    
    print_example(
        "5. Multi-GPU with More Data Workers",
        "./train_multigpu.sh 2 --batch_size 2 --num_workers 8",
        "Use more workers for faster data loading."
    )
    
    print_example(
        "6. Using torch.distributed.launch Directly",
        "python -m torch.distributed.launch --nproc_per_node=2 train_flow_matching.py --batch_size 2",
        "Alternative way to launch multi-GPU training."
    )
    
    print_example(
        "7. Using torchrun (PyTorch >= 1.10)",
        "torchrun --nproc_per_node=2 train_flow_matching.py --batch_size 2",
        "Modern PyTorch launcher for distributed training."
    )
    
    print_example(
        "8. Monitor GPU Usage While Training",
        "watch -n 1 nvidia-smi",
        "Run in a separate terminal to monitor GPU utilization."
    )
    
    print("\n" + "="*70)
    print("Tips:")
    print("="*70)
    print("• Effective batch size = batch_size × num_gpus")
    print("• Scale learning rate when using more GPUs")
    print("• Use --num_workers to speed up data loading")
    print("• Checkpoints are saved only on the main process")
    print("• All GPUs must have enough memory for the batch size specified")
    print("="*70)
    print("\nFor more information, see MULTI_GPU_TRAINING.md")
    print()


if __name__ == "__main__":
    main()
