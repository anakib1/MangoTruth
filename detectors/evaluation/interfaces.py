"""Evaluation interfaces for consistent API across training and standalone evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from detectors.data.datasets.interfaces import TextDatasetInterface
from detectors.interfaces import IDetector
from detectors.metrics import SplitConclusion


@dataclass
class EvaluationResult:
    """Unified result format for all evaluation operations."""
    
    # Core metrics
    conclusion: SplitConclusion
    
    # Raw data for further analysis
    predictions: np.ndarray  # Model predictions/probabilities
    true_labels: np.ndarray  # Ground truth labels (in model's label space)
    
    # Metadata
    dataset_info: Dict[str, Any]
    model_info: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    
    # Optional additional data
    per_sample_predictions: Optional[List[float]] = None
    confidence_scores: Optional[np.ndarray] = None
    misclassified_indices: Optional[List[int]] = None


class IEvaluator(ABC):
    """Interface for model evaluators.
    
    This interface provides a consistent API for evaluation that can be
    used both during training and for standalone model assessment.
    """
    
    @abstractmethod
    def evaluate(self, 
                model: IDetector,
                dataset: TextDatasetInterface,
                batch_size: int = 32,
                **kwargs) -> EvaluationResult:
        """Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate.
            dataset: Dataset to evaluate on.
            batch_size: Batch size for prediction.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            EvaluationResult with comprehensive evaluation data.
        """
        pass
    
    @abstractmethod
    def batch_evaluate(self,
                      model: IDetector,
                      datasets: Dict[str, TextDatasetInterface],
                      batch_size: int = 32,
                      **kwargs) -> Dict[str, EvaluationResult]:
        """Evaluate a model on multiple datasets.
        
        Args:
            model: Model to evaluate.
            datasets: Dictionary of dataset name to dataset.
            batch_size: Batch size for prediction.
            **kwargs: Additional evaluation parameters.
            
        Returns:
            Dictionary mapping dataset names to evaluation results.
        """
        pass


class IMetricsCalculator(ABC):
    """Interface for metrics calculation."""
    
    @abstractmethod
    def calculate_metrics(self,
                         true_labels: np.ndarray,
                         predictions: np.ndarray,
                         **kwargs) -> SplitConclusion:
        """Calculate classification metrics.
        
        Args:
            true_labels: Ground truth binary labels.
            predictions: Model predictions/probabilities.
            **kwargs: Additional metrics parameters.
            
        Returns:
            SplitConclusion with metrics and visualizations.
        """
        pass 