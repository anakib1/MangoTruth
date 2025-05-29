"""Factory for creating trainer instances.

This factory centralizes the creation of trainer instances based on
configuration and model type, reducing code duplication across scripts.
"""

from typing import Type, Dict, Any, Optional
from enum import Enum

from detectors.training.interfaces import ITrainer, ITrainable, TrainingConfig
from detectors.training.configs import HuggingFaceTrainingConfig, ClassicalMLTrainingConfig
from detectors.training.trainers import (
    HuggingFaceTrainer,
    PerplexityTrainer, 
    GhostbusterTrainer,
    BaseTrainer
)


class TrainerType(Enum):
    """Enumeration of available trainer types."""
    HUGGINGFACE = "huggingface"
    PERPLEXITY = "perplexity"
    GHOSTBUSTER = "ghostbuster"
    CLASSICAL_ML = "classical_ml"


class TrainerFactory:
    """Factory class for creating trainer instances."""
    
    _trainer_registry: Dict[TrainerType, Type[ITrainer]] = {
        TrainerType.HUGGINGFACE: HuggingFaceTrainer,
        TrainerType.PERPLEXITY: PerplexityTrainer,
        TrainerType.GHOSTBUSTER: GhostbusterTrainer,
    }
    
    @classmethod
    def create_trainer(cls,
                      trainer_type: TrainerType,
                      model: ITrainable,
                      config: TrainingConfig) -> ITrainer:
        """Create a trainer instance.
        
        Args:
            trainer_type: Type of trainer to create.
            model: Trainable model instance.
            config: Training configuration.
            
        Returns:
            Configured trainer instance.
            
        Raises:
            ValueError: If trainer type is not supported.
        """
        if trainer_type not in cls._trainer_registry:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
        
        trainer_class = cls._trainer_registry[trainer_type]
        return trainer_class(model, config)
    
    @classmethod
    def create_from_config_type(cls,
                               model: ITrainable,
                               config: TrainingConfig) -> ITrainer:
        """Create trainer based on configuration type.
        
        Args:
            model: Trainable model instance.
            config: Training configuration.
            
        Returns:
            Appropriate trainer instance.
        """
        if isinstance(config, HuggingFaceTrainingConfig):
            return cls.create_trainer(TrainerType.HUGGINGFACE, model, config)
        elif isinstance(config, ClassicalMLTrainingConfig):
            return cls.create_trainer(TrainerType.CLASSICAL_ML, model, config)
        else:
            # Default to base trainer
            return BaseTrainer(model, config)
    
    @classmethod
    def register_trainer(cls, trainer_type: TrainerType, trainer_class: Type[ITrainer]):
        """Register a new trainer type.
        
        Args:
            trainer_type: Type identifier for the trainer.
            trainer_class: Trainer class to register.
        """
        cls._trainer_registry[trainer_type] = trainer_class
    
    @classmethod
    def get_available_trainers(cls) -> Dict[str, str]:
        """Get list of available trainer types.
        
        Returns:
            Dictionary mapping trainer type names to descriptions.
        """
        return {
            trainer_type.value: trainer_class.__doc__ or "No description available"
            for trainer_type, trainer_class in cls._trainer_registry.items()
        } 