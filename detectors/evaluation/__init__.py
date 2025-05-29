"""Centralized evaluation module for model assessment.

This module provides a unified evaluation interface that can be used
both standalone and during training, eliminating duplication.
"""

from .evaluators import StandardEvaluator, EvaluationResult
from .metrics import MetricsCalculator
from .interfaces import IEvaluator

__all__ = [
    'StandardEvaluator',
    'EvaluationResult', 
    'MetricsCalculator',
    'IEvaluator'
] 