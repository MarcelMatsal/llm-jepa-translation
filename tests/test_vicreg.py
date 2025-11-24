"""
Test script for VICReg loss implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.losses import create_loss, list_available_losses

def test_vicreg_loss():
    """Test VICReg loss creation and computation."""
    print("="*80)
    print("Testing VICReg Loss")
    print("="*80)
    
    # List available losses
    print("\nAvailable losses:")
    for name in list_available_losses().keys():
        print(f"  - {name}")
    
    # Create VICReg loss
    config = {
        'type': 'vicreg',
        'sim_coeff': 25.0,
        'std_coeff': 25.0,
        'cov_coeff': 1.0,
        'eps': 0.0001
    }
    
    print(f"\nCreating VICReg loss with config: {config}")
    loss_fn, components = create_loss(config, embedding_dim=768)
    
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Components: {list(components.keys())}")
    
    # Create dummy embeddings
    batch_size = 32
    embedding_dim = 768
    
    z1 = torch.randn(batch_size, embedding_dim, requires_grad=True)
    z2 = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    print(f"\nTesting with embeddings:")
    print(f"  z1 shape: {z1.shape}")
    print(f"  z2 shape: {z2.shape}")
    
    # Compute loss
    loss, metrics = loss_fn.compute(z1, z2)
    
    print(f"\nResults:")
    print(f"  Total Loss: {loss.item():.6f}")
    print(f"  Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    
    # Test backward pass
    print(f"\nTesting backward pass...")
    loss.backward()
    print("  ✓ Backward pass successful")
    print(f"  ✓ z1 grad shape: {z1.grad.shape}")
    print(f"  ✓ z2 grad shape: {z2.grad.shape}")
    
    # Test with identical embeddings (should have low invariance loss, high variance loss)
    print(f"\nTesting with identical embeddings...")
    z_same = torch.randn(batch_size, embedding_dim)
    loss_same, metrics_same = loss_fn.compute(z_same, z_same.clone())
    print(f"  Invariance loss (should be ~0): {metrics_same['invariance_loss']:.6f}")
    print(f"  Total loss: {loss_same.item():.6f}")
    
    # Test with very different embeddings (should have high invariance loss)
    print(f"\nTesting with very different embeddings...")
    z_diff1 = torch.randn(batch_size, embedding_dim)
    z_diff2 = torch.randn(batch_size, embedding_dim) * 10  # Very different
    loss_diff, metrics_diff = loss_fn.compute(z_diff1, z_diff2)
    print(f"  Invariance loss (should be high): {metrics_diff['invariance_loss']:.6f}")
    print(f"  Total loss: {loss_diff.item():.6f}")
    
    # Test collapse prevention
    print(f"\nTesting collapse prevention...")
    # Create collapsed embeddings (all same value)
    z_collapsed = torch.ones(batch_size, embedding_dim) * 0.5
    loss_collapsed, metrics_collapsed = loss_fn.compute(z_collapsed, z_collapsed)
    print(f"  Variance loss (should be high): {metrics_collapsed['variance_loss']:.6f}")
    print(f"  Total loss: {loss_collapsed.item():.6f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)

if __name__ == '__main__':
    test_vicreg_loss()
