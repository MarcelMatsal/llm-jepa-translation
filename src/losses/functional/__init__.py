"""
Functional alignment losses (no architectural components required).
"""
from .sigreg import SIGRegLoss
from .infonce import InfoNCELoss

__all__ = ['SIGRegLoss', 'InfoNCELoss']
