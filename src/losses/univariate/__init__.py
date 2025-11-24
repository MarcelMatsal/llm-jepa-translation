"""
Univariate statistical tests for goodness-of-fit.
"""
from .base import UnivariateTest
from .epps_pulley import EppsPulley

__all__ = ['UnivariateTest', 'EppsPulley']
