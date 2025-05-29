from typing import Dict, Any, Union, List
import numpy as np
from tqdm import tqdm

from detectors.interfaces import IDetector
from detectors.metrics import ClassificationMetrics, SplitConclusion, ClassificationRepresentations
from detectors.utils.training import report_classification
from detectors.data.datasets.interfaces.dataset_interface import TextDatasetInterface
import logging

logger = logging.getLogger(__name__)

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
                        batch_size: int = 32) -> SplitConclusion:
        """
        Evaluate model on a dataset
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for prediction
            
        Returns:
            SplitConclusion containing metrics and visualizations
        """
        # Get texts and labels
        texts = [sample.output for sample in dataset]
        true_labels = [1 if sample.label == "ai" else 0 for sample in dataset]  # Assuming binary classification
        
        # Get predictions in batches
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i:i + batch_size]
            batch_preds = self._prepare_predictions(batch_texts)
            predictions.extend(batch_preds)
            
        predictions = np.array(predictions)
        
        # For binary classification, use the positive class probability
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1]
            
        # Calculate metrics
        return report_classification(true_labels, predictions)


def evaluate(model: IDetector, 
            dataset: TextDatasetInterface,
            batch_size: int = 32) -> SplitConclusion:
    """
    Convenience function to evaluate a model on a dataset
    
    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        batch_size: Batch size for prediction
        
    Returns:
        SplitConclusion containing metrics and visualizations
    """
    evaluator = ModelEvaluator(model)
    return evaluator.evaluate_dataset(dataset, batch_size) 