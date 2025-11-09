"""
Test script to verify memory optimization features work correctly.
This tests gradient checkpointing and block offloading without requiring a full model checkpoint.
"""

import torch
import sys
from comfywan import WanModel, WanVAE

def test_gradient_checkpointing():
    """Test gradient checkpointing for WanModel"""
    print("Testing gradient checkpointing for WanModel...")
    
    # Create a small model for testing
    model = WanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        in_dim=48,
        out_dim=48,
        dim=128,  # Smaller for testing
        ffn_dim=512,  # Smaller for testing
        num_heads=4,
        num_layers=2,  # Just 2 layers for testing
        text_len=512,
        text_dim=128,
        gradient_checkpointing=False,  # Start with it off
    )
    
    # Test enabling/disabling gradient checkpointing
    assert model.gradient_checkpointing == False, "Initial gradient checkpointing should be False"
    
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing == True, "Gradient checkpointing should be enabled"
    
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing == False, "Gradient checkpointing should be disabled"
    
    # Test forward pass with gradient checkpointing enabled
    model.enable_gradient_checkpointing()
    model.train()  # Set to training mode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dummy inputs
    batch_size = 1
    x = torch.randn(batch_size, 48, 4, 32, 32, device=device)
    timestep = torch.tensor([500.0], device=device)
    context = torch.randn(batch_size, 512, 128, device=device)
    
    # Forward pass
    try:
        output = model(x, timestep, context)
        print(f"  ✓ Forward pass with gradient checkpointing successful, output shape: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print("  ✓ Backward pass with gradient checkpointing successful")
        
    except Exception as e:
        print(f"  ✗ Error during forward/backward pass: {e}")
        return False
    
    print("✓ Gradient checkpointing tests passed for WanModel\n")
    return True


def test_block_offloading():
    """Test block offloading for WanModel"""
    print("Testing block offloading for WanModel...")
    
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        print("  ⚠ Skipping block offloading test (CUDA not available)")
        return True
    
    # Create a small model for testing
    model = WanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        in_dim=48,
        out_dim=48,
        dim=128,
        ffn_dim=512,
        num_heads=4,
        num_layers=2,
        text_len=512,
        text_dim=128,
        offload_to_cpu=False,
    )
    
    # Test enabling/disabling offloading
    assert model.offload_to_cpu == False, "Initial offloading should be False"
    
    model.enable_offloading()
    assert model.offload_to_cpu == True, "Offloading should be enabled"
    
    model.disable_offloading()
    assert model.offload_to_cpu == False, "Offloading should be disabled"
    
    # Test forward pass with offloading enabled
    model.enable_offloading()
    model.eval()  # Set to eval mode
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # Create dummy inputs
    batch_size = 1
    x = torch.randn(batch_size, 48, 4, 32, 32, device=device)
    timestep = torch.tensor([500.0], device=device)
    context = torch.randn(batch_size, 512, 128, device=device)
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(x, timestep, context)
        print(f"  ✓ Forward pass with block offloading successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"  ✗ Error during forward pass: {e}")
        return False
    
    print("✓ Block offloading tests passed for WanModel\n")
    return True


def test_vae_gradient_checkpointing():
    """Test gradient checkpointing for WanVAE"""
    print("Testing gradient checkpointing for WanVAE...")
    
    # Create a small VAE for testing
    vae = WanVAE(
        dim=32,  # Smaller for testing
        dec_dim=32,
        z_dim=4,
        dim_mult=[1, 2],  # Fewer layers
        num_res_blocks=1,  # Fewer blocks
        attn_scales=[],
        temperal_downsample=[True],
        dropout=0.0,
        gradient_checkpointing=False,
    )
    
    # Test enabling/disabling gradient checkpointing
    assert vae.gradient_checkpointing == False, "Initial gradient checkpointing should be False"
    
    vae.enable_gradient_checkpointing()
    assert vae.gradient_checkpointing == True, "Gradient checkpointing should be enabled"
    
    vae.disable_gradient_checkpointing()
    assert vae.gradient_checkpointing == False, "Gradient checkpointing should be disabled"
    
    # Test encode with gradient checkpointing enabled
    vae.enable_gradient_checkpointing()
    vae.train()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    
    # Create dummy input (small video)
    # Shape: [B, C, T, H, W]
    batch_size = 1
    x = torch.randn(batch_size, 12, 4, 64, 64, device=device)
    
    try:
        # Test encode
        latent = vae.encode(x)
        print(f"  ✓ Encode with gradient checkpointing successful, latent shape: {latent.shape}")
        
        # Test backward pass through encoder
        loss = latent.sum()
        loss.backward()
        print("  ✓ Backward pass through encoder with gradient checkpointing successful")
        
    except Exception as e:
        print(f"  ✗ Error during VAE encode: {e}")
        return False
    
    print("✓ Gradient checkpointing tests passed for WanVAE\n")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Memory Optimization Features")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Gradient Checkpointing (WanModel)", test_gradient_checkpointing()))
    results.append(("Block Offloading (WanModel)", test_block_offloading()))
    results.append(("Gradient Checkpointing (WanVAE)", test_vae_gradient_checkpointing()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
