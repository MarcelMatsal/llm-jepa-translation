"""
SIGReg: Sketched Isotropic Gaussian Regularization

Based on LeJEPA paper (Balestriero & LeCun, 2025):
https://arxiv.org/abs/2511.08544

Official implementation:
https://github.com/rbalestr-lab/lejepa

This implementation uses the official LeJEPA modular architecture:
- Univariate test: EppsPulley (tests 1D samples against N(0,1))
- Multivariate wrapper: SlicingUnivariateTest (projects to 1D via random slicing)
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from ..base import FunctionalLoss
from ..registry import register_loss
from ..univariate import EppsPulley
from ..multivariate import SlicingUnivariateTest


@register_loss('sigreg')
class SIGRegLoss(FunctionalLoss):
    """
    SIGReg: Sketched Isotropic Gaussian Regularization.
    
    Enforces embeddings to follow N(0, I) distribution using:
    - Random slicing (Cramér-Wold theorem)
    - Epps-Pulley characteristic function test
    
    This is the official LeJEPA architecture with two layers:
    1. Univariate test (EppsPulley): Tests 1D projections against N(0,1)
    2. Multivariate wrapper (SlicingUnivariateTest): Projects D-dim data to 1D
    
    Args:
        config: Configuration dictionary with keys:
            - num_slices: Number of random 1D projections (default: 256)
            - num_points: Number of points for Epps-Pulley test (default: 17)
            - t_max: Maximum integration point (default: 3.0)
            - normalize_embeddings: Whether to normalize before testing (default: True)
            - reduction: How to aggregate across slices ('mean' or 'sum', default: 'mean')
    
    Example:
        >>> config = {
        ...     'type': 'sigreg',
        ...     'num_slices': 256,
        ...     'num_points': 17,
        ...     'normalize_embeddings': True
        ... }
        >>> loss_fn = SIGRegLoss(config)
        >>> z1 = torch.randn(32, 768)  # batch_size=32, dim=768
        >>> z2 = torch.randn(32, 768)
        >>> loss, metrics = loss_fn.compute(z1, z2)
    """
    
    def __init__(self, config: Dict):
        self.num_slices = config.get('num_slices', 256)
        self.num_points = config.get('num_points', 17)
        self.t_max = config.get('t_max', 3.0)
        self.normalize_embeddings = config.get('normalize_embeddings', True)
        self.reduction = config.get('reduction', 'mean')
        
        # Create univariate test (tests 1D samples against N(0,1))
        univariate_test = EppsPulley(
            t_max=self.t_max,
            n_points=self.num_points,
            integration='trapezoid'
        )
        
        # Wrap with multivariate slicing (projects D-dim to 1D)
        self.test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=self.num_slices,
            reduction=self.reduction,
            sampler='gaussian',
            clip_value=None
        )
    
    def compute(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SIGReg loss.
        
        Args:
            z1: Embeddings from view 1 (batch_size, dim)
            z2: Embeddings from view 2 (batch_size, dim)
            
        Returns:
            loss: SIGReg loss (scalar)
            metrics: Dictionary with loss and statistics
        """
        # Combine both views for testing
        # In cross-lingual case: both lang1 and lang2 CLS tokens
        embeddings = torch.cat([z1, z2], dim=0)  # (2*batch_size, dim)
        
        # Optional: normalize embeddings
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=-1)
        
        # Apply SIGReg test
        # Input: (N, D) where N = 2*batch_size, D = embedding_dim
        # Output: scalar (if reduction='mean')
        loss = self.test(embeddings)
        
        # Compute additional metrics
        with torch.no_grad():
            # Embedding statistics
            emb_mean = embeddings.mean(dim=0).norm().item()
            emb_std = embeddings.std(dim=0).mean().item()
            
            # Sample a few projections for monitoring
            sample_size = min(16, self.num_slices)
            A_sample = torch.randn(embeddings.size(-1), sample_size, 
                                  device=embeddings.device, dtype=embeddings.dtype)
            A_sample = F.normalize(A_sample, dim=0)
            proj_sample = embeddings @ A_sample
            proj_mean = proj_sample.mean().item()
            proj_std = proj_sample.std().item()
        
        metrics = {
            'alignment_loss': loss.item(),
            'sigreg_loss': loss.item(),
            'embedding_mean_norm': emb_mean,
            'embedding_std': emb_std,
            'projection_mean': proj_mean,
            'projection_std': proj_std,
            'num_slices': self.num_slices
        }
        
        return loss, metrics
