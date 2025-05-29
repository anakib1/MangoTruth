"""Centralized metrics calculation for evaluation.

This module consolidates metrics calculation logic that was previously
scattered across utils/training.py and evaluation modules.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

from detectors.metrics import ClassificationMetrics, SplitConclusion, ClassificationRepresentations
from detectors.evaluation.interfaces import IMetricsCalculator
from detectors.utils.training import (
    tpr_at_fpr_threshold, 
    safe_roc, 
    draw_classification, 
    calculate_classification
)

logger = logging.getLogger(__name__)


class MetricsCalculator(IMetricsCalculator):
    """Standard metrics calculator for classification tasks."""
    
    def __init__(self, 
                 threshold: float = 0.5,
                 include_visualizations: bool = True,
                 fpr_thresholds: Optional[list] = None):
        """Initialize the metrics calculator.
        
        Args:
            threshold: Classification threshold for binary predictions.
            include_visualizations: Whether to generate ROC/confusion matrix plots.
            fpr_thresholds: FPR thresholds for TPR calculation (default: [0.01, 0.1]).
        """
        self.threshold = threshold
        self.include_visualizations = include_visualizations
        self.fpr_thresholds = fpr_thresholds or [0.01, 0.1]
    
    def calculate_metrics(self,
                         true_labels: np.ndarray,
                         predictions: np.ndarray,
                         **kwargs) -> SplitConclusion:
        """Calculate comprehensive classification metrics.
        
        Args:
            true_labels: Ground truth binary labels (0/1).
            predictions: Model predictions/probabilities.
            **kwargs: Additional parameters (threshold, etc.).
            
        Returns:
            SplitConclusion with metrics and optional visualizations.
        """
        # Use provided threshold or default
        threshold = kwargs.get('threshold', self.threshold)
        include_viz = kwargs.get('include_visualizations', self.include_visualizations)
        
        logger.debug(f"Calculating metrics with threshold={threshold}, "
                    f"include_viz={include_viz}")
        
        # Calculate standard metrics
        metrics = calculate_classification(true_labels, predictions)
        
        # Add custom TPR at FPR metrics
        fpr, tpr, auc = safe_roc(true_labels, predictions)
        for fpr_thresh in self.fpr_thresholds:
            tpr_value = tpr_at_fpr_threshold(fpr, tpr, target_fpr=fpr_thresh)
            # Add to metrics (extend ClassificationMetrics if needed)
            setattr(metrics, f'tpr_at_{int(fpr_thresh*100)}_percent_fpr', tpr_value)
        
        # Generate visualizations if requested
        representations = None
        if include_viz:
            try:
                representations = draw_classification(true_labels, predictions)
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
                representations = ClassificationRepresentations(
                    roc_curve=None,
                    clf_report=None
                )
        
        return SplitConclusion(
            metrics=metrics,
            representations=representations
        )
    
    def calculate_batch_metrics(self,
                               results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               **kwargs) -> Dict[str, SplitConclusion]:
        """Calculate metrics for multiple datasets.
        
        Args:
            results: Dictionary mapping dataset names to (true_labels, predictions).
            **kwargs: Additional parameters passed to calculate_metrics.
            
        Returns:
            Dictionary mapping dataset names to SplitConclusion.
        """
        batch_metrics = {}
        for dataset_name, (true_labels, predictions) in results.items():
            logger.info(f"Calculating metrics for dataset: {dataset_name}")
            batch_metrics[dataset_name] = self.calculate_metrics(
                true_labels, predictions, **kwargs
            )
        return batch_metrics


class MultiClassMetricsCalculator(IMetricsCalculator):
    """Metrics calculator for multi-class classification."""
    
    def __init__(self, 
                 average_method: str = 'macro',
                 include_visualizations: bool = True):
        """Initialize the multi-class metrics calculator.
        
        Args:
            average_method: Averaging method ('macro', 'micro', 'weighted').
            include_visualizations: Whether to generate confusion matrix plots.
        """
        self.average_method = average_method
        self.include_visualizations = include_visualizations
    
    def calculate_metrics(self,
                         true_labels: np.ndarray,
                         predictions: np.ndarray,
                         **kwargs) -> SplitConclusion:
        """Calculate multi-class classification metrics.
        
        Note: This is a placeholder for future multi-class support.
        Currently converts to binary classification.
        """
        logger.warning("Multi-class metrics not fully implemented. "
                      "Converting to binary classification.")
        
        # Convert to binary for now
        predicted_labels = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        binary_true = (predicted_labels == true_labels).astype(int)
        binary_pred = np.max(predictions, axis=1) if predictions.ndim > 1 else predictions
        
        # Use standard calculator
        standard_calc = MetricsCalculator(
            include_visualizations=self.include_visualizations
        )
        return standard_calc.calculate_metrics(binary_true, binary_pred, **kwargs)


class MetricsCalculatorFactory:
    """Factory for creating appropriate metrics calculators."""
    
    @staticmethod
    def create_calculator(task_type: str = 'binary', **kwargs) -> IMetricsCalculator:
        """Create appropriate metrics calculator for the task.
        
        Args:
            task_type: Type of classification task ('binary', 'multiclass').
            **kwargs: Additional parameters for the calculator.
            
        Returns:
            Configured metrics calculator.
        """
        if task_type == 'binary':
            return MetricsCalculator(**kwargs)
        elif task_type == 'multiclass':
            return MultiClassMetricsCalculator(**kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    @staticmethod
    def create_for_model(model, **kwargs) -> IMetricsCalculator:
        """Create calculator based on model characteristics.
        
        Args:
            model: Model to create calculator for.
            **kwargs: Additional parameters.
            
        Returns:
            Configured metrics calculator.
        """
        # Determine task type from model
        try:
            labels = model.get_labels()
            task_type = 'binary' if len(labels) == 2 else 'multiclass'
        except:
            logger.warning("Could not determine model task type, defaulting to binary")
            task_type = 'binary'
        
        return MetricsCalculatorFactory.create_calculator(task_type, **kwargs) 