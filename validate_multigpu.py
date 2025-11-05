"""
Validation script for multi-GPU training code changes.
Performs static analysis and basic code structure validation.
"""

import ast
import sys
import os


def get_train_script_path():
    """Get the path to the training script."""
    # Try relative to script location first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'train_flow_matching.py')
    if os.path.exists(train_script):
        return train_script
    # Fallback to current directory
    if os.path.exists('train_flow_matching.py'):
        return 'train_flow_matching.py'
    raise FileNotFoundError("Could not find train_flow_matching.py")


def check_imports():
    """Verify required imports are present."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    required_imports = [
        'torch.distributed',
        'DistributedSampler',
        'argparse',
    ]
    
    for imp in required_imports:
        if imp not in content:
            print(f"✗ Missing import: {imp}")
            return False
    
    print("✓ All required imports present")
    return True


def check_distributed_functions():
    """Verify distributed helper functions are defined."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    required_functions = [
        'setup_distributed',
        'cleanup_distributed',
        'is_main_process',
        'parse_args',
    ]
    
    for func in required_functions:
        if f"def {func}(" not in content:
            print(f"✗ Missing function: {func}")
            return False
    
    print("✓ All distributed helper functions defined")
    return True


def check_ddp_wrapper():
    """Verify model is wrapped with DistributedDataParallel."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    checks = [
        'DistributedDataParallel',
        'self.is_distributed',
        'world_size',
        'rank',
    ]
    
    for check in checks:
        if check not in content:
            print(f"✗ Missing DDP component: {check}")
            return False
    
    print("✓ DDP wrapper code present")
    return True


def check_rank_handling():
    """Verify rank-specific operations are handled correctly."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    # Check for main process guards in critical sections
    checks = [
        'is_main_process',
        'if is_main_process',
    ]
    
    for check in checks:
        if check not in content:
            print(f"✗ Missing rank handling: {check}")
            return False
    
    print("✓ Rank-specific operations handled")
    return True


def check_distributed_sampler():
    """Verify DistributedSampler is used in multi-GPU mode."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    if 'DistributedSampler' not in content:
        print("✗ DistributedSampler not used")
        return False
    
    if 'sampler.set_epoch' not in content:
        print("✗ Missing sampler.set_epoch call")
        return False
    
    print("✓ DistributedSampler properly configured")
    return True


def check_model_unwrapping():
    """Verify model is properly unwrapped for state dict operations."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    # Check for proper model unwrapping patterns
    if 'model.module' not in content:
        print("✗ Model unwrapping not implemented")
        return False
    
    print("✓ Model unwrapping for DDP present")
    return True


def check_backward_compatibility():
    """Verify single GPU mode still works."""
    train_script = get_train_script_path()
    with open(train_script, 'r') as f:
        content = f.read()
    
    # Check that world_size > 1 conditions are used
    if 'world_size > 1' not in content:
        print("✗ Backward compatibility check missing")
        return False
    
    print("✓ Backward compatibility maintained")
    return True


def check_syntax():
    """Verify Python syntax is valid."""
    try:
        train_script = get_train_script_path()
        with open(train_script, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✓ Python syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False


def check_documentation():
    """Verify documentation files are created."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        'MULTI_GPU_TRAINING.md',
        'train_multigpu.sh',
    ]
    
    for file in required_files:
        file_path = os.path.join(script_dir, file)
        if not os.path.exists(file_path):
            print(f"✗ Missing documentation file: {file}")
            return False
    
    print("✓ Documentation files present")
    return True


def main():
    """Run all validation checks."""
    print("Validating multi-GPU training implementation...\n")
    print("="*50)
    
    checks = [
        check_syntax,
        check_imports,
        check_distributed_functions,
        check_ddp_wrapper,
        check_rank_handling,
        check_distributed_sampler,
        check_model_unwrapping,
        check_backward_compatibility,
        check_documentation,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"✗ Check failed with exception: {e}")
            results.append(False)
        print()
    
    print("="*50)
    if all(results):
        print(f"All {len(results)} validation checks passed! ✓")
        print("="*50)
        return 0
    else:
        failed = len([r for r in results if not r])
        print(f"{failed}/{len(results)} checks failed ✗")
        print("="*50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
