"""
Functional alignment losses (no architectural components required).
"""
from .sigreg import SIGRegLoss
from .vicreg import VICRegLoss

__all__ = ['SIGRegLoss', 'VICRegLoss']
