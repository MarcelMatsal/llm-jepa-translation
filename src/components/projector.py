"""
Projector network for SSL losses.

Based on SimCLR and SSL Cookbook recommendations.
"""
import torch.nn as nn


class Projector(nn.Module):
    """
    MLP projector for SSL methods.
    
    Standard 2-3 layer MLP with BatchNorm and ReLU.
    Maps encoder representations to projection space where loss is computed.
    
    Benefits (from SSL Cookbook):
    - Prevents dimensional collapse (+20% performance)
    - Handles noisy augmentations
    - Separates pretext task from downstream task
    
    Args:
        input_dim: Input dimension (encoder output)
        hidden_dim: Hidden layer dimension (default: 2048)
        output_dim: Output projection dimension (default: 128)
        num_layers: Number of layers (2 or 3, default: 2)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        assert num_layers in [2, 3], "num_layers must be 2 or 3"
        
        if num_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        else:  # num_layers == 3
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Projected tensor (batch_size, output_dim)
        """
        return self.net(x)
