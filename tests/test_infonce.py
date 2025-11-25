"""
Test script for InfoNCE loss implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.losses import create_loss, list_available_losses

def test_infonce_loss():
    """Test InfoNCE loss creation and computation."""
    print("="*80)
    print("Testing InfoNCE Loss")
    print("="*80)
    
    # List available losses
    print("\nAvailable losses:")
    for name in list_available_losses().keys():
        print(f"  - {name}")
    
    # Test 1: InfoNCE without projector
    print("\n" + "="*80)
    print("Test 1: InfoNCE without projector")
    print("="*80)
    
    config = {
        'type': 'infonce',
        'temperature': 0.07,
        'normalize': True,
        'use_projector': False
    }
    
    print(f"\nConfig: {config}")
    loss_fn, components = create_loss(config, embedding_dim=768)
    
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Components: {list(components.keys())}")
    
    # Create dummy embeddings
    batch_size = 32
    embedding_dim = 768
    
    z1 = torch.randn(batch_size, embedding_dim, requires_grad=True)
    z2 = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    print(f"\nInput shapes:")
    print(f"  z1: {z1.shape}")
    print(f"  z2: {z2.shape}")
    
    # Compute loss
    loss, metrics = loss_fn.compute(z1, z2)
    
    print(f"\nResults:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Metrics:")
    for key, value in sorted(metrics.items()):
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
    
    # Test 2: InfoNCE with projector
    print("\n" + "="*80)
    print("Test 2: InfoNCE with projector")
    print("="*80)
    
    config_proj = {
        'type': 'infonce',
        'temperature': 0.07,
        'normalize': True,
        'use_projector': True,
        'projector_hidden_dim': 2048,
        'projector_output_dim': 128,
        'projector_num_layers': 2
    }
    
    print(f"\nConfig: {config_proj}")
    loss_fn_proj, components_proj = create_loss(config_proj, embedding_dim=768)
    
    print(f"Loss function: {loss_fn_proj.__class__.__name__}")
    print(f"Components: {list(components_proj.keys())}")
    if 'projector' in components_proj:
        print(f"  Projector: {components_proj['projector']}")
    
    # Create new embeddings
    z1_proj = torch.randn(batch_size, embedding_dim, requires_grad=True)
    z2_proj = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    # Compute loss
    loss_proj, metrics_proj = loss_fn_proj.compute(z1_proj, z2_proj)
    
    print(f"\nResults with projector:")
    print(f"  Loss: {loss_proj.item():.6f}")
    print(f"  Positive sim: {metrics_proj['positive_sim']:.6f}")
    print(f"  Negative sim: {metrics_proj['negative_sim']:.6f}")
    print(f"  Contrastive accuracy: {metrics_proj['contrastive_accuracy']:.6f}")
    
    # Test backward
    loss_proj.backward()
    print(f"\n  ✓ Backward pass successful with projector")
    
    # Test 3: Verify contrastive behavior
    print("\n" + "="*80)
    print("Test 3: Verify contrastive behavior")
    print("="*80)
    
    # Create similar pairs (should have high positive sim)
    z1_similar = torch.randn(16, 768, requires_grad=True)
    z2_similar = z1_similar + 0.1 * torch.randn(16, 768)  # Very similar
    
    loss_similar, metrics_similar = loss_fn.compute(z1_similar, z2_similar)
    
    print(f"\nSimilar pairs:")
    print(f"  Loss: {loss_similar.item():.6f}")
    print(f"  Positive sim: {metrics_similar['positive_sim']:.6f}")
    print(f"  Negative sim: {metrics_similar['negative_sim']:.6f}")
    print(f"  Accuracy: {metrics_similar['contrastive_accuracy']:.6f}")
    
    # Verify positive > negative
    assert metrics_similar['positive_sim'] > metrics_similar['negative_sim'], \
        "Positive similarity should be higher than negative!"
    print(f"\n  ✓ Positive sim > Negative sim (as expected)")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)

if __name__ == '__main__':
    test_infonce_loss()
