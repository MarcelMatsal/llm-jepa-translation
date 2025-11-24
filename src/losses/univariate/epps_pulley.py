"""
Epps-Pulley univariate test for normality.

Based on LeJEPA implementation:
https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/univariate/epps_pulley.py
"""
import torch
import torch.nn as nn
from typing import Optional

from .base import UnivariateTest


class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.
    
    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.
    
    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt
    
    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.
    
    Args:
        t_max (float): Maximum integration point. Default: 3.0
        n_points (int): Number of integration points (must be odd). Default: 17
        integration (str): Integration method ('trapezoid'). Default: 'trapezoid'
    
    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0)
        weights (torch.Tensor): Precomputed integration weights
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values
    
    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed
        - Contributions from -t are implicitly added via doubled weights
    """
    
    def __init__(
        self,
        t_max: float = 3.0,
        n_points: int = 17,
        integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        
        self.integration = integration
        self.n_points = n_points
        self.t_max = t_max
        
        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        
        # Trapezoidal rule weights
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at endpoints (t=0 and t=t_max)
        
        # Precompute φ(t) = exp(-t²/2) for standard normal
        phi = torch.exp(-0.5 * t**2)
        self.register_buffer("phi", phi)
        
        # Combine weights and phi for efficiency
        self.register_buffer("weights", weights * phi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Epps-Pulley test statistic.
        
        Args:
            x: Input samples of shape (*, N, K) where:
               * = batch dimensions
               N = number of samples
               K = number of slices
        
        Returns:
            Test statistics of shape (*, K)
        """
        N = x.size(-2)
        
        # Ensure buffers are on same device as input
        if self.t.device != x.device:
            self.t = self.t.to(x.device)
            self.phi = self.phi.to(x.device)
            self.weights = self.weights.to(x.device)
        
        # Compute cos/sin for all integration points
        # x: (*, N, K)
        # t: (n_points,)
        # x_t: (*, N, K, n_points)
        x_t = x.unsqueeze(-1) * self.t
        
        cos_vals = torch.cos(x_t)  # (*, N, K, n_points)
        sin_vals = torch.sin(x_t)  # (*, N, K, n_points)
        
        # Mean across samples (dimension -3 is N)
        cos_mean = cos_vals.mean(-3)  # (*, K, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, K, n_points)
        
        # Compute error from standard normal
        # cos_mean should match phi, sin_mean should be 0
        err = (cos_mean - self.phi)**2 + sin_mean**2  # (*, K, n_points)
        
        # Weighted integration (dot product with weights)
        # err: (*, K, n_points)
        # weights: (n_points,)
        # result: (*, K)
        test_stat = (err * self.weights).sum(-1) * N * self.world_size
        
        return test_stat
