"""
Utility functions for configuration and logging.
"""
import os
from typing import Dict, Any
import torch


def get_device(device_str: str = 'cuda') -> torch.device:
    """Get device, auto-detect if 'cuda'."""
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def setup_directories(save_dir: str):
    """Create output directories."""
    os.makedirs(save_dir, exist_ok=True)

