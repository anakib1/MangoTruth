"""Centralized evaluator implementations.

This module provides evaluators that can be used both during training
and for standalone evaluation, eliminating code duplication.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from detectors.interfaces import IDetector
from detectors.data.datasets.interfaces import TextDatasetInterface
from detectors.evaluation.interfaces import IEvaluator, EvaluationResult
from detectors.evaluation.metrics import MetricsCalculatorFactory

logger = logging.getLogger(__name__)

# Label mapping constants for common AI/human classification variants
AI_RELATED_LABELS = ['ai', 'llm', 'machine', 'artificial']
HUMAN_RELATED_LABELS = ['human', 'person', 'real']


class StandardEvaluator(IEvaluator):
    """Standard evaluator for text classification models.
    
    This evaluator consolidates the logic from the old ModelEvaluator
    and provides a consistent interface for both training and standalone use.
    """
    
    def __init__(self, 
                 metrics_calculator=None,
                 label_mapping_strategy: str = 'auto'):
        """Initialize the evaluator.
        
        Args:
            metrics_calculator: Optional custom metrics calculator.
            label_mapping_strategy: Strategy for mapping dataset labels to model labels.
        """
        self.metrics_calculator = metrics_calculator
        self.label_mapping_strategy = label_mapping_strategy
    
    def _create_label_mapping(self, 
                             dataset: TextDatasetInterface, 
                             model: IDetector) -> Dict[str, int]:
        """Create mapping from dataset labels to model label indices.
        
        Args:
            dataset: Dataset to extract labels from.
            model: Model to get label space from.
            
        Returns:
            Dictionary mapping dataset label strings to model indices.
        """
        dataset_labels = dataset.get_labels()
        model_labels = model.get_labels()
        
        logger.debug(f"Dataset labels: {dataset_labels}")
        logger.debug(f"Model labels: {model_labels}")
        
        label_mapping = {}
        
        # Try exact matches first
        for dataset_label in dataset_labels:
            if dataset_label in model_labels:
                label_mapping[dataset_label] = model_labels.index(dataset_label)
            elif dataset_label.lower() in [label.lower() for label in model_labels]:
                # Case-insensitive match
                for i, model_label in enumerate(model_labels):
                    if model_label.lower() == dataset_label.lower():
                        label_mapping[dataset_label] = i
                        break
        
        # Handle remaining labels with semantic mapping
        for dataset_label in dataset_labels:
            if dataset_label not in label_mapping:
                if dataset_label.lower() in AI_RELATED_LABELS:
                    # Look for AI-related labels in model
                    for i, model_label in enumerate(model_labels):
                        if model_label.lower() in AI_RELATED_LABELS:
                            label_mapping[dataset_label] = i
                            logger.info(f"Mapped '{dataset_label}' to model label '{model_label}' (index {i})")
                            break
                elif dataset_label.lower() in HUMAN_RELATED_LABELS:
                    # Look for human-related labels in model
                    for i, model_label in enumerate(model_labels):
                        if model_label.lower() in HUMAN_RELATED_LABELS:
                            label_mapping[dataset_label] = i
                            logger.info(f"Mapped '{dataset_label}' to model label '{model_label}' (index {i})")
                            break
                
                # If still not found, use default mapping
                if dataset_label not in label_mapping:
                    label_mapping[dataset_label] = 0
                    logger.warning(f"Using default index 0 for unmapped label '{dataset_label}'")
        
        return label_mapping
    
    def _prepare_predictions(self, 
                           model: IDetector, 
                           texts: List[str],
                           batch_size: int = 32) -> np.ndarray:
        """Get model predictions for a list of texts.
        
        Args:
            model: Model to use for prediction.
            texts: List of texts to predict on.
            batch_size: Batch size for prediction.
            
        Returns:
            Array of predictions.
        """
        try:
            logger.debug(f"Getting predictions for {len(texts)} texts")
            
            # Get predictions in batches
            all_predictions = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
                batch_texts = texts[i:i + batch_size]
                batch_preds = model.batch_predict(batch_texts)
                
                # Ensure predictions are numpy array
                if not isinstance(batch_preds, np.ndarray):
                    batch_preds = np.array(batch_preds)
                
                all_predictions.append(batch_preds)
            
            return np.vstack(all_predictions)
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def _convert_to_binary_classification(self,
                                        predictions: np.ndarray,
                                        true_labels: np.ndarray,
                                        model_labels: List[str]) -> tuple:
        """Convert predictions and labels to binary classification format.
        
        Args:
            predictions: Model predictions (potentially multi-class).
            true_labels: True label indices.
            model_labels: List of model label names.
            
        Returns:
            Tuple of (binary_predictions, binary_true_labels).
        """
        if len(model_labels) == 2:
            # Binary classification
            if predictions.shape[1] == 2:
                # Use probability of positive class (typically AI/LLM)
                positive_idx = 1  # default
                for i, label in enumerate(model_labels):
                    if label.lower() in AI_RELATED_LABELS:
                        positive_idx = i
                        break
                
                binary_predictions = predictions[:, positive_idx]
                binary_true_labels = (true_labels == positive_idx).astype(int)
            else:
                binary_predictions = predictions.flatten()
                binary_true_labels = true_labels
        else:
            # Multi-class: convert to binary "correct/incorrect"
            logger.warning("Multi-class evaluation with binary metrics")
            predicted_labels = np.argmax(predictions, axis=1)
            binary_true_labels = (predicted_labels == true_labels).astype(int)
            binary_predictions = np.max(predictions, axis=1)
        
        return binary_predictions, binary_true_labels
    
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
        logger.info(f"Evaluating model {model.__class__.__name__} on dataset with {len(dataset)} samples")
        
        # Extract texts and create label mapping
        texts = [sample.output for sample in dataset]
        label_mapping = self._create_label_mapping(dataset, model)
        
        # Map dataset labels to model label indices
        true_labels = []
        for sample in dataset:
            if sample.label in label_mapping:
                true_labels.append(label_mapping[sample.label])
            else:
                logger.warning(f"Unknown label '{sample.label}' in dataset sample")
                true_labels.append(0)  # Default to first label
        
        true_labels = np.array(true_labels)
        
        # Get model predictions
        predictions = self._prepare_predictions(model, texts, batch_size)
        
        # Convert to binary classification format for metrics
        model_labels = model.get_labels()
        binary_predictions, binary_true_labels = self._convert_to_binary_classification(
            predictions, true_labels, model_labels
        )
        
        # Calculate metrics
        if self.metrics_calculator is None:
            self.metrics_calculator = MetricsCalculatorFactory.create_for_model(model)
        
        conclusion = self.metrics_calculator.calculate_metrics(
            binary_true_labels, binary_predictions, **kwargs
        )
        
        # Create comprehensive result
        result = EvaluationResult(
            conclusion=conclusion,
            predictions=binary_predictions,
            true_labels=binary_true_labels,
            dataset_info={
                'name': getattr(dataset, 'name', 'unknown'),
                'size': len(dataset),
                'labels': dataset.get_labels(),
                'label_mapping': label_mapping
            },
            model_info={
                'class': model.__class__.__name__,
                'labels': model_labels,
                'parameters': getattr(model, 'get_num_parameters', lambda: 'unknown')()
            },
            evaluation_config={
                'batch_size': batch_size,
                'metrics_calculator': self.metrics_calculator.__class__.__name__,
                **kwargs
            }
        )
        
        logger.info(f"Evaluation completed. Accuracy: {conclusion.metrics.accuracy:.4f}, "
                   f"F1: {conclusion.metrics.f1:.4f}, AUC: {conclusion.metrics.auc:.4f}")
        
        return result
    
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
        results = {}
        for dataset_name, dataset in datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")
            results[dataset_name] = self.evaluate(
                model, dataset, batch_size, **kwargs
            )
        return results


# Convenience function for backward compatibility
def evaluate_model(model: IDetector, 
                  dataset: TextDatasetInterface,
                  batch_size: int = 32,
                  **kwargs) -> EvaluationResult:
    """Convenience function to evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataset: Dataset to evaluate on.
        batch_size: Batch size for prediction.
        **kwargs: Additional evaluation parameters.
        
    Returns:
        EvaluationResult with comprehensive evaluation data.
    """
    evaluator = StandardEvaluator()
    return evaluator.evaluate(model, dataset, batch_size, **kwargs) 