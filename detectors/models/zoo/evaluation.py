from typing import Dict, Any, Union, List, Tuple
import numpy as np
from tqdm import tqdm

from detectors.interfaces import IDetector
from detectors.metrics import ClassificationMetrics, SplitConclusion, ClassificationRepresentations
from detectors.utils.training import report_classification
from detectors.data.datasets.interfaces.dataset_interface import TextDatasetInterface
import logging

logger = logging.getLogger(__name__)

# Label mapping constants for common AI/human classification variants
AI_RELATED_LABELS = ['ai', 'llm', 'machine', 'artificial']
HUMAN_RELATED_LABELS = ['human', 'person', 'real']

class ModelEvaluator:
    """Evaluator for text classification models"""

    def __init__(self, model: IDetector):
        self.model = model
        self.labels = model.get_labels()
        logger.info(f"Initializing ModelEvaluator for model: {model.__class__.__name__}")

    def _prepare_predictions(self, texts: List[str]) -> np.ndarray:
        """Get model predictions for a list of texts"""
        try:
            logger.info("Getting batch predictions from model")
            predictions = self.model.batch_predict(texts)
            # Ensure predictions are numpy array on CPU
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            return predictions
        except Exception as e:
            logger.error(f"Error in batch_predict: {str(e)}")
            raise

    def evaluate_dataset(self, 
                        dataset: TextDatasetInterface,
                        batch_size: int = 32) -> Tuple[SplitConclusion, np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for prediction
            
        Returns:
            Tuple containing:
            - SplitConclusion: Metrics and visualizations
            - predictions: Array of prediction scores used for metrics computation
            - true_labels: Array of true label indices (mapped to model's label space)
        """
        # Get texts and labels
        texts = [sample.output for sample in dataset]
        
        # Get all unique labels from dataset and model
        dataset_labels = dataset.get_labels()
        model_labels = self.model.get_labels()
        
        logger.info(f"Dataset labels: {dataset_labels}")
        logger.info(f"Model labels: {model_labels}")
        
        # Create label mapping from dataset labels to model label indices
        label_to_idx = {}
        for dataset_label in dataset_labels:
            # Try to find exact match first
            if dataset_label in model_labels:
                label_to_idx[dataset_label] = model_labels.index(dataset_label)
            else:
                # Try case-insensitive match
                lower_model_labels = [label.lower() for label in model_labels]
                if dataset_label.lower() in lower_model_labels:
                    label_to_idx[dataset_label] = lower_model_labels.index(dataset_label.lower())
                else:
                    logger.warning(f"Dataset label '{dataset_label}' not found in model labels {model_labels}")
                    # Default mapping for common cases
                    if dataset_label.lower() in AI_RELATED_LABELS:
                        # Look for AI-related labels in model
                        for i, model_label in enumerate(model_labels):
                            if model_label.lower() in AI_RELATED_LABELS:
                                label_to_idx[dataset_label] = i
                                break
                    elif dataset_label.lower() in HUMAN_RELATED_LABELS:
                        # Look for human-related labels in model
                        for i, model_label in enumerate(model_labels):
                            if model_label.lower() in HUMAN_RELATED_LABELS:
                                label_to_idx[dataset_label] = i
                                break
                    
                    # If still not found, use first available index
                    if dataset_label not in label_to_idx:
                        label_to_idx[dataset_label] = 0
                        logger.warning(f"Using default index 0 for label '{dataset_label}'")
        
        # Map dataset labels to model label indices
        true_labels = []
        for sample in dataset:
            if sample.label in label_to_idx:
                true_labels.append(label_to_idx[sample.label])
            else:
                logger.warning(f"Unknown label '{sample.label}' in dataset sample")
                true_labels.append(0)  # Default to first label
        
        true_labels = np.array(true_labels)
        
        # Get predictions in batches
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i:i + batch_size]
            batch_preds = self._prepare_predictions(batch_texts)
            predictions.extend(batch_preds)
            
        predictions = np.array(predictions)
        # Handle different prediction scenarios
        if len(model_labels) == 2:
            # Binary classification - use positive class probability for metrics
            # Assume the "positive" class is AI/LLM (index 1) or the second class
            if predictions.shape[1] == 2:
                # Use the probability of the positive class (typically AI/LLM)
                positive_idx = 1  # default
                for i, label in enumerate(model_labels):
                    if label.lower() in AI_RELATED_LABELS:
                        positive_idx = i
                        break
                predictions_for_metrics = predictions[:, positive_idx]
                # Convert true labels to binary (1 if matches positive class, 0 otherwise)
                true_labels_binary = (true_labels == positive_idx).astype(int)
            else:
                predictions_for_metrics = predictions.flatten()
                true_labels_binary = true_labels
        else:
            # Multi-class classification - use argmax for now
            # Note: The current metrics framework is designed for binary classification
            # For multi-class, we could modify to use macro/micro averaging
            logger.warning("Multi-class evaluation with binary metrics - using argmax predictions")
            predictions_for_metrics = np.max(predictions, axis=1)
            # Convert to binary by checking if prediction matches true label
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels_binary = (predicted_labels == true_labels).astype(int)
            predictions_for_metrics = np.where(predicted_labels == true_labels, 
                                             np.max(predictions, axis=1), 
                                             1 - np.max(predictions, axis=1))
            
        # Calculate metrics
        metrics = report_classification(true_labels_binary, predictions_for_metrics)
        return metrics, predictions_for_metrics, true_labels


def evaluate(model: IDetector, 
            dataset: TextDatasetInterface,
            batch_size: int = 32) -> Tuple[SplitConclusion, np.ndarray, np.ndarray]:
    """
    Convenience function to evaluate a model on a dataset
    
    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        batch_size: Batch size for prediction
        
    Returns:
        Tuple containing:
        - SplitConclusion: Metrics and visualizations
        - predictions: Array of prediction scores used for metrics computation
        - true_labels: Array of true label indices (mapped to model's label space)
    """
    evaluator = ModelEvaluator(model)
    return evaluator.evaluate_dataset(dataset, batch_size) 