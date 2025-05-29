"""Training module for various detection models."""

from .interfaces import ITrainable, ITrainer, TrainingConfig, TrainingResult
from .configs import BaseTrainingConfig, HuggingFaceTrainingConfig, ClassicalMLTrainingConfig

# Import trainers lazily to avoid circular imports
def _get_trainers():
    from .trainers import (
        BaseTrainer,
        HuggingFaceTrainer,
        PerplexityTrainer,
        PerplexityTrainingConfig,
        GhostbusterTrainer,
        GhostbusterTrainingConfig
    )
    return {
        'BaseTrainer': BaseTrainer,
        'HuggingFaceTrainer': HuggingFaceTrainer,
        'PerplexityTrainer': PerplexityTrainer,
        'PerplexityTrainingConfig': PerplexityTrainingConfig,
        'GhostbusterTrainer': GhostbusterTrainer,
        'GhostbusterTrainingConfig': GhostbusterTrainingConfig
    }

def __getattr__(name: str):
    """Lazily import trainers when accessed."""
    trainers = _get_trainers()
    if name in trainers:
        return trainers[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Core interfaces
    'ITrainable',
    'ITrainer',
    'TrainingConfig',
    'TrainingResult',
    
    # Configuration classes
    'BaseTrainingConfig',
    'HuggingFaceTrainingConfig',
    'ClassicalMLTrainingConfig',
    
    # Trainer classes (imported lazily)
    'BaseTrainer',
    'HuggingFaceTrainer',
    'PerplexityTrainer',
    'PerplexityTrainingConfig',
    'GhostbusterTrainer',
    'GhostbusterTrainingConfig'
] 