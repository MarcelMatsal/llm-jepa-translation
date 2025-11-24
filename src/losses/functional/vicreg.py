"""
VICReg: Variance-Invariance-Covariance Regularization

Based on the paper:
"VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
Adrien Bardes, Jean Ponce, Yann LeCun (2021)
https://arxiv.org/abs/2105.04906

Official implementation:
https://github.com/facebookresearch/vicreg

VICReg explicitly prevents collapse through three terms:
1. Invariance: MSE between embeddings from different views
2. Variance: Maintains variance along each dimension (prevents dimensional collapse)
3. Covariance: Decorrelates dimensions (reduces redundancy)

Key advantages:
- No negative pairs needed
- No momentum encoder needed
- No stop-gradient needed
- Explicit collapse prevention
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from ..base import FunctionalLoss
from ..registry import register_loss


@register_loss('vicreg')
class VICRegLoss(FunctionalLoss):
    """
    VICReg: Variance-Invariance-Covariance Regularization.
    
    Combines three loss terms:
    1. Invariance (sim): MSE between view pairs
    2. Variance (std): Encourages high variance along each dimension
    3. Covariance (cov): Decorrelates different dimensions
    
    Loss = λ * invariance + μ * variance + ν * covariance
    
    Args:
        config: Configuration dictionary with keys:
            - sim_coeff: Weight for invariance loss (default: 25.0)
            - std_coeff: Weight for variance loss (default: 25.0)
            - cov_coeff: Weight for covariance loss (default: 1.0)
            - eps: Small constant for numerical stability (default: 0.0001)
    
    Example:
        >>> config = {
        ...     'type': 'vicreg',
        ...     'sim_coeff': 25.0,
        ...     'std_coeff': 25.0,
        ...     'cov_coeff': 1.0
        ... }
        >>> loss_fn = VICRegLoss(config)
        >>> z1 = torch.randn(32, 768)
        >>> z2 = torch.randn(32, 768)
        >>> loss, metrics = loss_fn.compute(z1, z2)
    """
    
    def __init__(self, config: Dict):
        # Loss coefficients (from paper)
        self.sim_coeff = config.get('sim_coeff', 25.0)
        self.std_coeff = config.get('std_coeff', 25.0)
        self.cov_coeff = config.get('cov_coeff', 1.0)
        self.eps = config.get('eps', 0.0001)
    
    def compute(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss.
        
        Args:
            z1: Embeddings from view 1 (batch_size, dim)
            z2: Embeddings from view 2 (batch_size, dim)
            
        Returns:
            loss: VICReg loss (scalar)
            metrics: Dictionary with loss components and statistics
        """
        batch_size = z1.size(0)
        num_features = z1.size(1)
        
        # 1. Invariance loss: MSE between paired embeddings
        # Encourages same-image views to have similar representations
        invariance_loss = F.mse_loss(z1, z2)
        
        # Center the embeddings (zero mean)
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        # 2. Variance loss: Maintains variance along each dimension
        # Prevents dimensional collapse (all embeddings having same value in a dimension)
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        
        # Hinge loss: penalize std < 1, encourage std >= 1
        variance_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2
        
        # 3. Covariance loss: Decorrelates different dimensions
        # Reduces redundancy by encouraging zero covariance between dimensions
        
        # Compute covariance matrices
        # cov = (X^T @ X) / (N - 1)
        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        
        # Off-diagonal elements (we want these to be zero)
        # Diagonal elements are variances (handled by variance loss)
        off_diag_cov_z1 = self._off_diagonal(cov_z1).pow(2).sum() / num_features
        off_diag_cov_z2 = self._off_diagonal(cov_z2).pow(2).sum() / num_features
        
        covariance_loss = off_diag_cov_z1 + off_diag_cov_z2
        
        # Combined loss
        loss = (
            self.sim_coeff * invariance_loss +
            self.std_coeff * variance_loss +
            self.cov_coeff * covariance_loss
        )
        
        # Compute metrics
        with torch.no_grad():
            # Embedding statistics
            z1_mean_norm = z1.mean(dim=0).norm().item()
            z2_mean_norm = z2.mean(dim=0).norm().item()
            z1_std_mean = std_z1.mean().item()
            z2_std_mean = std_z2.mean().item()
            
            # Cosine similarity between views
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            cosine_sim = (z1_norm * z2_norm).sum(dim=-1).mean().item()
        
        metrics = {
            'alignment_loss': loss.item(),
            'vicreg_loss': loss.item(),
            'invariance_loss': invariance_loss.item(),
            'variance_loss': variance_loss.item(),
            'covariance_loss': covariance_loss.item(),
            'z1_std_mean': z1_std_mean,
            'z2_std_mean': z2_std_mean,
            'cosine_sim': cosine_sim,
            'sim_coeff': self.sim_coeff,
            'std_coeff': self.std_coeff,
            'cov_coeff': self.cov_coeff
        }
        
        return loss, metrics
    
    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """
        Extract off-diagonal elements of a square matrix.
        
        Args:
            x: Square matrix (n, n)
            
        Returns:
            Flattened off-diagonal elements
        """
        n, m = x.shape
        assert n == m, "Matrix must be square"
        # Flatten, remove last element, reshape to (n-1, n+1), take column [1:], flatten
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
