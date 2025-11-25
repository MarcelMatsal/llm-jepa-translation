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
            'type': 'infonce',
            'temperature': 0.07,
            'use_projector': True
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
    
    # Special handling for InfoNCE projector (optional component)
    if loss_type == 'infonce' and config.get('use_projector', False):
        from src.components import Projector
        required_components['projector'] = Projector
    
    # Instantiate components
    components = {}
    for comp_name, comp_class in required_components.items():
        # Get component-specific config, default to empty dict
        comp_config = config.get(comp_name, {})
        
        # Add embedding_dim to component config
        comp_config['input_dim'] = embedding_dim
        
        # Add default projector params if not specified
        if comp_name == 'projector':
            comp_config.setdefault('hidden_dim', config.get('projector_hidden_dim', 2048))
            comp_config.setdefault('output_dim', config.get('projector_output_dim', 128))
            comp_config.setdefault('num_layers', config.get('projector_num_layers', 2))
        
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
from .functional import SIGRegLoss


# Export public API
__all__ = [
    'AlignmentLoss',
    'FunctionalLoss', 
    'ArchitecturalLoss',
    'register_loss',
    'create_loss',
    'list_available_losses',
    'SIGRegLoss'
]

