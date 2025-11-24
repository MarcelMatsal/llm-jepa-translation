"""
Loss registry and factory for alignment losses.
"""
from typing import Dict, Type, Tuple
import torch.nn as nn

from .base import AlignmentLoss, FunctionalLoss, ArchitecturalLoss
from .registry import LOSS_REGISTRY, register_loss, list_available_losses


def create_loss(
    config: Dict,
    embedding_dim: int
) -> Tuple[AlignmentLoss, Dict[str, nn.Module]]:
    """
    Factory function to create a loss and its required components.
    
    Args:
        config: Loss configuration dictionary with 'type' key
        embedding_dim: Dimension of input embeddings from the encoder
        
    Returns:
        loss: Instantiated loss function
        components: Dictionary of components to add to the model
        
    Example:
        config = {
            'type': 'sigreg',
            'num_slices': 1024,
            'num_points': 17
        }
        loss, components = create_loss(config, embedding_dim=768)
    """
    loss_type = config.get('type', 'infonce')
    
    if loss_type not in LOSS_REGISTRY:
        available = ', '.join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Available losses: {available}"
        )
    
    loss_class = LOSS_REGISTRY[loss_type]
    
    # Get required components for this loss
    required_components = loss_class.get_required_components()
    
    # Instantiate components
    components = {}
    for comp_name, comp_class in required_components.items():
        # Get component-specific config, default to empty dict
        comp_config = config.get(comp_name, {})
        
        # Add embedding_dim to component config
        comp_config['input_dim'] = embedding_dim
        
        # Instantiate component
        components[comp_name] = comp_class(**comp_config)
    
    # Create loss instance
    if issubclass(loss_class, FunctionalLoss):
        # Functional losses don't need components
        loss = loss_class(config)
    else:
        # Architectural losses need components
        loss = loss_class(config, components)
    
    return loss, components


# Import functional losses to register them
# This must come AFTER the factory definition to avoid circular imports
from .functional import SIGRegLoss, VICRegLoss


# Export public API
__all__ = [
    'AlignmentLoss',
    'FunctionalLoss', 
    'ArchitecturalLoss',
    'register_loss',
    'create_loss',
    'list_available_losses',
    'SIGRegLoss',
    'VICRegLoss'
]

