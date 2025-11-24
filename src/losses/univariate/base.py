"""
Base class for univariate statistical tests.
"""
import torch
import torch.nn as nn


class UnivariateTest(nn.Module):
    """
    Base class for univariate goodness-of-fit tests.
    
    Tests whether 1D samples come from a standard Gaussian N(0, 1).
    """
    
    @property
    def world_size(self):
        """Get world size for distributed training."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute test statistic.
        
        Args:
            x: Input samples of shape (*, N, K) where:
               * = batch dimensions
               N = number of samples
               K = number of slices/features to test
        
        Returns:
            Test statistics of shape (*, K)
        """
        raise NotImplementedError
