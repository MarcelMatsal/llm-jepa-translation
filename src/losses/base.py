"""
Base classes for alignment losses.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Callable
import torch
import torch.nn as nn


class AlignmentLoss(ABC):
    """
    Base class for all alignment losses.
    
    All losses must implement:
    - compute(): Calculate loss from embeddings
    - get_required_components(): Specify needed architectural components
    """
    
    @abstractmethod
    def compute(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute alignment loss between two views.
        
        Args:
            z1: Embeddings from view 1 (batch_size, embedding_dim)
            z2: Embeddings from view 2 (batch_size, embedding_dim)
            **kwargs: Method-specific additional arguments
            
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
                Must include 'alignment_loss' key
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_required_components() -> Dict[str, type]:
        """
        Return required architectural components.
        
        Returns:
            Dictionary mapping component name to component class
            Example: {'projection_head': ProjectionHead}
            Return empty dict if no components needed
        """
        pass
    
    def get_update_hooks(self) -> List[Callable[[], None]]:
        """
        Return functions to call after optimizer.step().
        
        Used for:
        - EMA updates (momentum encoders)
        - Queue/support set updates
        
        Returns:
            List of callables that take no arguments
        """
        return []


class FunctionalLoss(AlignmentLoss):
    """
    Base class for losses that don't require architectural components.
    
    These losses are pure functions of the embeddings:
    - VICReg
    - Barlow Twins
    - SIGReg
    - InfoNCE (basic version)
    """
    
    @staticmethod
    def get_required_components() -> Dict[str, type]:
        """Functional losses don't need components."""
        return {}


class ArchitecturalLoss(AlignmentLoss):
    """
    Base class for losses that require architectural components.
    
    These losses need additional modules:
    - SimCLR (projection head)
    - MoCo (projection head + queue + momentum encoder)
    - BYOL (projection head + predictor + momentum encoder)
    - SimSiam (projection head + predictor)
    - NNCLR (projection head + support set)
    """
    
    def __init__(self, config: Dict, components: Dict[str, nn.Module]):
        """
        Initialize architectural loss with components.
        
        Args:
            config: Loss configuration dictionary
            components: Pre-instantiated components from factory
        """
        self.config = config
        self.components = components
