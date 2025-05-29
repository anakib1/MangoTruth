"""Base trainer implementation for extensibility.

This module provides a base trainer class that can be extended
for different training approaches, including non-deep-learning methods.
"""

from abc import abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json
from pathlib import Path

from detectors.data.datasets.interfaces import TextDatasetInterface
from detectors.training.interfaces import ITrainer, ITrainable, TrainingResult, TrainingConfig
from detectors.metrics import Conclusion
from detectors.evaluation import StandardEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class BaseTrainer(ITrainer):
    """Base trainer implementation with common functionality.
    
    This class provides common functionality for all trainers and can be
    extended for specific training approaches.
    """
    
    def __init__(self, model: ITrainable, config: TrainingConfig):
        """Initialize the trainer.
        
        Args:
            model: Trainable model instance.
            config: Training configuration.
        """
        self.model = model
        self.config = config
        self._training_history: Dict[str, List[float]] = {}
        self._best_metrics: Dict[str, float] = {}
        self._evaluator = StandardEvaluator()
        self._setup()
    
    def _setup(self) -> None:
        """Perform any additional setup needed by the trainer."""
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save initial configuration
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log metrics to training history.
        
        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step.
            prefix: Prefix for metric names (e.g., "train_", "val_").
        """
        for name, value in metrics.items():
            key = f"{prefix}{name}"
            if key not in self._training_history:
                self._training_history[key] = []
            self._training_history[key].append(value)
            
            # Track best metrics
            if "loss" in name:
                # For loss, lower is better
                if key not in self._best_metrics or value < self._best_metrics[key]:
                    self._best_metrics[key] = value
            else:
                # For other metrics, higher is better
                if key not in self._best_metrics or value > self._best_metrics[key]:
                    self._best_metrics[key] = value
    
    def _should_stop_early(self, current_metric: float, patience_counter: int) -> bool:
        """Check if training should stop early.
        
        Args:
            current_metric: Current metric value.
            patience_counter: Number of epochs without improvement.
            
        Returns:
            True if training should stop.
        """
        if not self.config.early_stopping:
            return False
        
        return patience_counter >= self.config.early_stopping_patience
    
    def _create_training_result(self,
                               start_time: datetime,
                               end_time: datetime,
                               train_conclusion: Conclusion,
                               validation_conclusion: Optional[Conclusion] = None,
                               test_conclusion: Optional[Conclusion] = None,
                               best_checkpoint_path: Optional[str] = None) -> TrainingResult:
        """Create a TrainingResult object.
        
        Args:
            start_time: Training start time.
            end_time: Training end time.
            train_conclusion: Training set conclusion.
            validation_conclusion: Optional validation set conclusion.
            test_conclusion: Optional test set conclusion.
            best_checkpoint_path: Path to best checkpoint.
            
        Returns:
            TrainingResult object.
        """
        return TrainingResult(
            run_id=self.config.run_id,
            model_name=getattr(self.config, 'model_name', 'unknown'),
            trainer_type=self.__class__.__name__,
            start_time=start_time,
            end_time=end_time,
            training_duration_seconds=(end_time - start_time).total_seconds(),
            train_conclusion=train_conclusion,
            validation_conclusion=validation_conclusion,
            test_conclusion=test_conclusion,
            model_weights=self.model.store_weights(),
            best_checkpoint_path=best_checkpoint_path,
            training_history=self._training_history,
            config=self.config,
            metadata=self._get_metadata()
        )
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata for the training result.
        
        Can be overridden by subclasses to add specific metadata.
        
        Returns:
            Dictionary of metadata.
        """
        return {
            'best_metrics': self._best_metrics,
            'trainer_class': self.__class__.__name__,
        }
    
    def evaluate(self, dataset: TextDatasetInterface, **kwargs) -> Conclusion:
        """Evaluate the model on a dataset using centralized evaluator.
        
        Args:
            dataset: Dataset to evaluate on.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            Conclusion with evaluation metrics.
        """
        logger.info(f"Evaluating model on dataset with {len(dataset)} samples")
        
        # Use centralized evaluator
        evaluation_result = self._evaluator.evaluate(
            model=self.model, 
            dataset=dataset, 
            **kwargs
        )
        
        # Convert EvaluationResult back to Conclusion for compatibility
        # Note: This maintains backward compatibility while using centralized evaluation
        return evaluation_result.conclusion.train_conclusion if hasattr(evaluation_result.conclusion, 'train_conclusion') else evaluation_result.conclusion

    def evaluate_comprehensive(self, dataset: TextDatasetInterface, **kwargs) -> EvaluationResult:
        """Perform comprehensive evaluation using centralized evaluator.
        
        This method provides access to the full EvaluationResult with additional
        metadata and analysis capabilities.
        
        Args:
            dataset: Dataset to evaluate on.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            EvaluationResult with comprehensive evaluation data.
        """
        return self._evaluator.evaluate(
            model=self.model,
            dataset=dataset,
            **kwargs
        )

    @abstractmethod
    def train(self,
              train_dataset: TextDatasetInterface,
              validation_dataset: Optional[TextDatasetInterface] = None,
              test_dataset: Optional[TextDatasetInterface] = None) -> TrainingResult:
        """Train the model.
        
        Must be implemented by subclasses.
        
        Args:
            train_dataset: Training dataset.
            validation_dataset: Optional validation dataset.
            test_dataset: Optional test dataset.
            
        Returns:
            TrainingResult containing metrics and artifacts.
        """
        pass
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save a training checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint.
            epoch: Current epoch number.
            metrics: Current metrics.
        """
        # Save model
        self.model.save_checkpoint(checkpoint_path)
        
        # Save trainer state
        state_path = Path(checkpoint_path) / "trainer_state.json"
        state = {
            'epoch': epoch,
            'metrics': metrics,
            'training_history': self._training_history,
            'best_metrics': self._best_metrics,
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to load checkpoint from.
            
        Returns:
            Dictionary containing trainer state.
        """
        # Load model
        self.model.load_checkpoint(checkpoint_path)
        
        # Load trainer state
        state_path = Path(checkpoint_path) / "trainer_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self._training_history = state.get('training_history', {})
            self._best_metrics = state.get('best_metrics', {})
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state
        
        return {}
    
    def save_model(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model.
        """
        self.model.save_checkpoint(path)
        
        # Also save training history and metadata
        metadata_path = Path(path) / "training_metadata.json"
        metadata = {
            'training_history': self._training_history,
            'best_metrics': self._best_metrics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            'metadata': self._get_metadata()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, path: str) -> None:
        """Load a trained model.
        
        Args:
            path: Path to load the model from.
        """
        self.model.load_checkpoint(path)
        
        # Load training metadata if available
        metadata_path = Path(path) / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self._training_history = metadata.get('training_history', {})
            self._best_metrics = metadata.get('best_metrics', {}) 