"""
Multivariate distribution test using random slicing.

Based on LeJEPA implementation:
https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/multivariate/slicing.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class SlicingUnivariateTest(nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.
    
    This module extends univariate statistical tests to multivariate data by projecting
    samples onto random 1D directions (slices) and aggregating univariate test statistics
    across all projections. The approach is based on the Cramér-Wold theorem.
    
    The test projects multivariate samples x ∈ ℝᴰ onto random unit vectors:
        x_projected = x @ A
    
    where A ∈ ℝᴰˣᴷ contains K normalized random direction vectors. A univariate
    test is then applied to each of the K projected samples, and results are aggregated.
    
    Args:
        univariate_test (nn.Module): A univariate test module that accepts
            (*, N, K) tensors and returns (*, K) test statistics
        num_slices (int): Number of random 1D projections (slices) to use
        reduction (str): How to aggregate statistics across slices:
            - 'mean': Return the average statistic across all slices
            - 'sum': Return the sum of statistics across all slices
            - None: Return individual statistics for each slice
            Default: 'mean'
        sampler (str): Random sampling method for projection directions:
            - 'gaussian': Sample from standard normal distribution
            Default: 'gaussian'
        clip_value (float, optional): Minimum threshold for test statistics.
            Values below this are clipped to zero. Default: None
    
    Shape:
        - Input: (*, N, D) where * is batch dimensions, N is samples, D is features
        - Output: Scalar if reduction='mean'/'sum', (*, num_slices) if reduction=None
    
    Example:
        >>> from src.losses.univariate import EppsPulley
        >>> from src.losses.multivariate import SlicingUnivariateTest
        >>>
        >>> # Create univariate test
        >>> univariate_test = EppsPulley(t_max=3.0, n_points=17)
        >>>
        >>> # Wrap with slicing for multivariate testing
        >>> test = SlicingUnivariateTest(
        ...     univariate_test=univariate_test,
        ...     num_slices=256,
        ...     reduction='mean'
        ... )
        >>>
        >>> # Test multivariate samples
        >>> samples = torch.randn(64, 768)  # 64 samples, 768 dimensions
        >>> statistic = test(samples)
    """
    
    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int,
        reduction: Optional[Literal['mean', 'sum']] = 'mean',
        sampler: str = 'gaussian',
        clip_value: Optional[float] = None
    ):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self.reduction = reduction
        self.sampler = sampler
        self.clip_value = clip_value
        
        # Global step counter for deterministic random seed
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply slicing-based multivariate test.
        
        Args:
            x: Input samples of shape (*, N, D) where:
               * = batch dimensions
               N = number of samples
               D = feature dimension
        
        Returns:
            Test statistic (scalar if reduction is set, else (*, num_slices))
        """
        # x: (*, N, D)
        D = x.size(-1)
        
        # Generate random projection directions
        # A: (D, num_slices)
        if self.sampler == 'gaussian':
            A = torch.randn(D, self.num_slices, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")
        
        # Normalize columns to unit vectors
        A = F.normalize(A, dim=0)
        
        # Project samples onto random directions
        # x: (*, N, D) @ A: (D, K) = (*, N, K)
        x_projected = x @ A
        
        # Apply univariate test to all K slices
        # Input: (*, N, K)
        # Output: (*, K)
        stats = self.univariate_test(x_projected)
        
        # Optional clipping
        if self.clip_value is not None:
            stats = torch.clamp(stats, min=self.clip_value)
        
        # Aggregate across slices
        if self.reduction == 'mean':
            return stats.mean()
        elif self.reduction == 'sum':
            return stats.sum()
        else:
            return stats
