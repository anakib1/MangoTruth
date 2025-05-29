"""Training factories for centralized object creation.

This module provides factory classes to create trainers and models
with standardized configurations and reduced code duplication.
"""

from .trainer_factory import TrainerFactory, TrainerType

__all__ = ['TrainerFactory', 'TrainerType'] 