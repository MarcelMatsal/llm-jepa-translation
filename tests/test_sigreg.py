"""
Test script for SIGReg loss implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.losses import create_loss, list_available_losses

def test_sigreg_loss():
    """Test SIGReg loss creation and computation."""
    print("="*80)
    print("Testing SIGReg Loss")
    print("="*80)
    
    # List available losses
    print("\nAvailable losses:")
    for name in list_available_losses().keys():
        print(f"  - {name}")
    
    # Create SIGReg loss
    config = {
        'type': 'sigreg',
        'num_slices': 128,  # Smaller for testing
        'num_points': 17,
        'normalize_embeddings': True
    }
    
    print(f"\nCreating SIGReg loss with config: {config}")
    loss_fn, components = create_loss(config, embedding_dim=768)
    
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Components: {list(components.keys())}")
    
    # Create dummy embeddings
    batch_size = 16
    embedding_dim = 768
    
    z1 = torch.randn(batch_size, embedding_dim, requires_grad=True)
    z2 = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    print(f"\nTesting with embeddings:")
    print(f"  z1 shape: {z1.shape}")
    print(f"  z2 shape: {z2.shape}")
    
    # Compute loss
    loss, metrics = loss_fn.compute(z1, z2)
    
    print(f"\nResults:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Test backward pass
    print(f"\nTesting backward pass...")
    loss.backward()
    print("  ✓ Backward pass successful")
    print(f"  ✓ z1 grad shape: {z1.grad.shape}")
    print(f"  ✓ z2 grad shape: {z2.grad.shape}")
    
    # Test with normalized embeddings (should be closer to Gaussian)
    print(f"\nTesting with normalized embeddings...")
    z1_norm = torch.randn(batch_size, embedding_dim)
    z2_norm = torch.randn(batch_size, embedding_dim)
    
    loss_norm, metrics_norm = loss_fn.compute(z1_norm, z2_norm)
    print(f"  Loss (normalized): {loss_norm.item():.6f}")
    print(f"  Loss (original): {loss.item():.6f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)

if __name__ == '__main__':
    test_sigreg_loss()
