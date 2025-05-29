"""Core training interfaces.

This module defines the abstract interfaces for trainable models and training
infrastructure, ensuring a consistent API across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from detectors.data.datasets.interfaces import TextDatasetInterface
from detectors.metrics import Conclusion


@dataclass
class TrainingConfig:
    """Base configuration for model training.
    
    This class defines the common configuration parameters
    that all training implementations should support.
    """
    # Basic training parameters
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = 0.1
    
    # Validation settings
    validation_split: float = 0.2
    validation_strategy: str = "epoch"  # "epoch", "steps", "no"
    validation_steps: Optional[int] = None
    
    # Logging and checkpointing
    logging_steps: int = 100
    save_strategy: str = "epoch"  # "epoch", "steps", "no"
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = 3
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001
    
    # Other settings
    seed: int = 42
    output_dir: str = "./training_output"
    run_id: UUID = field(default_factory=uuid4)
    
    # Additional framework-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            if isinstance(value, UUID):
                result[field_info.name] = str(value)
            else:
                result[field_info.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration values.
            
        Returns:
            New TrainingConfig instance.
        """
        # Handle UUID conversion
        if 'run_id' in data and isinstance(data['run_id'], str):
            data = data.copy()
            data['run_id'] = UUID(data['run_id'])
        
        return cls(**data)


@dataclass
class TrainingResult:
    """Result of a training run.
    
    This class encapsulates all the information produced by a training run,
    including metrics, model weights, and metadata.
    """
    # Required fields (no defaults)
    run_id: UUID
    model_name: str
    trainer_type: str
    start_time: datetime
    end_time: datetime
    training_duration_seconds: float
    train_conclusion: 'Conclusion'
    model_weights: bytes
    
    # Optional fields (with defaults)
    validation_conclusion: Optional['Conclusion'] = None
    test_conclusion: Optional['Conclusion'] = None
    best_checkpoint_path: Optional[str] = None
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    config: Optional[TrainingConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ITrainable(ABC):
    """Interface for trainable models.
    
    This interface extends the base detector interface with methods
    required for training. Models implementing this interface can be
    trained using the training infrastructure.
    """
    
    @abstractmethod
    def train_mode(self, mode: bool = True) -> None:
        """Set the model to training or evaluation mode.
        
        Args:
            mode: True for training mode, False for evaluation mode.
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> Dict[str, Any]:
        """Get the trainable parameters of the model.
        
        Returns:
            Dictionary of parameter name to parameter object.
        """
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update model parameters.
        
        Args:
            parameters: Dictionary of parameter name to new values.
        """
        pass
    
    @abstractmethod
    def compute_loss(self, inputs: Dict[str, Any], labels: Any) -> Tuple[Any, Dict[str, Any]]:
        """Compute the loss for given inputs and labels.
        
        Args:
            inputs: Model inputs (e.g., tokenized text).
            labels: Ground truth labels.
            
        Returns:
            Tuple of (loss, auxiliary_outputs) where auxiliary_outputs
            may contain logits, attention weights, etc.
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint.
        
        Args:
            path: Path to load the checkpoint from.
        """
        pass


class ITrainer(ABC):
    """Interface for model trainers.
    
    This interface defines the contract for trainer implementations
    that handle the training loop and optimization process.
    """
    
    @abstractmethod
    def __init__(self, model: ITrainable, config: TrainingConfig):
        """Initialize the trainer.
        
        Args:
            model: Trainable model instance.
            config: Training configuration.
        """
        self.model = model
        self.config = config
    
    @abstractmethod
    def train(self, 
              train_dataset: TextDatasetInterface,
              validation_dataset: Optional[TextDatasetInterface] = None,
              test_dataset: Optional[TextDatasetInterface] = None) -> TrainingResult:
        """Train the model.
        
        Args:
            train_dataset: Training dataset.
            validation_dataset: Optional validation dataset.
            test_dataset: Optional test dataset.
            
        Returns:
            TrainingResult containing metrics and artifacts.
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataset: TextDatasetInterface) -> Conclusion:
        """Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on.
            
        Returns:
            Conclusion with evaluation metrics.
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model.
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model.
        
        Args:
            path: Path to load the model from.
        """
        pass 