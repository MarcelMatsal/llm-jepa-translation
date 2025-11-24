"""
Loss registry for alignment losses.
"""
from typing import Dict, Type

# Global loss registry
LOSS_REGISTRY: Dict[str, Type] = {}


def register_loss(name: str):
    """
    Decorator to register a loss function in the global registry.
    
    Args:
        name: Name to register the loss under (used in config files)
        
    Example:
        @register_loss('sigreg')
        class SIGRegLoss(FunctionalLoss):
            ...
    """
    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def list_available_losses() -> Dict[str, Type]:
    """
    Get all registered losses.
    
    Returns:
        Dictionary mapping loss names to loss classes
    """
    return LOSS_REGISTRY.copy()
