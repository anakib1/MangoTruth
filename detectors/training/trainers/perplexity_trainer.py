"""Trainer for perplexity-based models."""

import time
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from detectors.training.interfaces import ITrainer, TrainingResult
from detectors.training.configs import BaseTrainingConfig
from detectors.training.trainers.base_trainer import BaseTrainer
from detectors.models.zoo.trainable_implementations import TrainablePerplexityModel
from detectors.data.datasets import Dataset
from detectors.utils.math import safe_sigmoid
from detectors.utils.training import calculate_classification


class PerplexityTrainingConfig(BaseTrainingConfig):
    """Configuration for perplexity model training."""
    
    def __init__(
        self,
        model_handle: str = 'babbage-002',
        validation_split: float = 0.3,
        batch_size: int = 4,
        test_split: float = 0.0,  # Use validation as test if no separate test set
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_handle = model_handle
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.test_split = test_split
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        config_dict = super().to_dict()
        config_dict.update({
            'model_handle': self.model_handle,
            'test_split': self.test_split
        })
        return config_dict


class PerplexityTrainer(BaseTrainer):
    """Trainer for perplexity-based models."""
    
    def __init__(self, model: TrainablePerplexityModel, config: PerplexityTrainingConfig):
        super().__init__(model, config)
        self.config: PerplexityTrainingConfig = config
    
    def train(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> TrainingResult:
        """Train the perplexity model."""
        from datetime import datetime
        
        start_time = datetime.now()
        
        # Convert dataset to texts and labels
        train_texts = [sample.output for sample in train_dataset]
        train_labels = [1 if sample.label == 'human' else 0 for sample in train_dataset]
        
        # Compute perplexities in batches
        perplexities = self._compute_perplexities(train_texts)
        
        # Split into train/validation if validation not provided
        if validation_dataset is None:
            X_train, X_val, y_train, y_val = train_test_split(
                perplexities, train_labels, 
                test_size=self.config.validation_split,
                stratify=train_labels,
                random_state=42
            )
        else:
            X_train, y_train = perplexities, train_labels
            val_texts = [sample.output for sample in validation_dataset]
            val_labels = [1 if sample.label == 'human' else 0 for sample in validation_dataset]
            X_val = self._compute_perplexities(val_texts)
            y_val = val_labels
        
        # Train threshold
        self.model.perplexity_threshold = self._train_threshold(X_train, y_train)
        
        # Train scaling factor
        self.model.scaling_factor = self._train_scaling_factor(
            X_train, y_train, self.model.perplexity_threshold
        )
        
        # Get predictions
        y_train_pred = safe_sigmoid(
            self.model.scaling_factor * (np.array(X_train) - self.model.perplexity_threshold)
        )
        y_val_pred = safe_sigmoid(
            self.model.scaling_factor * (np.array(X_val) - self.model.perplexity_threshold)
        )
        
        # Calculate metrics
        train_conclusion = calculate_classification(y_train, y_train_pred)
        val_conclusion = calculate_classification(y_val, y_val_pred)
        
        # Handle test set if provided
        test_conclusion = None
        if test_dataset is not None:
            test_texts = [sample.output for sample in test_dataset]
            test_labels = [1 if sample.label == 'human' else 0 for sample in test_dataset]
            X_test = self._compute_perplexities(test_texts)
            y_test_pred = safe_sigmoid(
                self.model.scaling_factor * (np.array(X_test) - self.model.perplexity_threshold)
            )
            test_conclusion = calculate_classification(test_labels, y_test_pred)
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        return TrainingResult(
            run_id=self.config.run_id,
            model_name="perplexity",
            trainer_type=self.__class__.__name__,
            start_time=start_time,
            end_time=end_time,
            training_duration_seconds=training_duration,
            train_conclusion=train_conclusion,
            model_weights=self.model.store_weights(),
            validation_conclusion=val_conclusion,
            test_conclusion=test_conclusion,
            best_checkpoint_path=None,
            config=self.config,
            metadata={
                'perplexity_threshold': self.model.perplexity_threshold,
                'scaling_factor': self.model.scaling_factor,
                'model_handle': self.model.model_handle
            }
        )
    
    def _compute_perplexities(self, texts):
        """Compute perplexities for texts."""
        from tqdm.auto import tqdm
        
        perplexities = []
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Computing perplexities"):
            batch_texts = texts[i:i + self.config.batch_size]
            for text in batch_texts:
                if self.model.model is not None:
                    perplexity = self.model.predict_llm(text)
                else:
                    perplexity = self.model.predict_openai(text)
                perplexities.append(perplexity)
        
        # Handle NaN values
        perplexities = np.array(perplexities)
        nan_mask = np.isnan(perplexities)
        if np.any(nan_mask):
            mean_val = np.nanmean(perplexities)
            perplexities[nan_mask] = mean_val
        
        return perplexities
    
    def _train_threshold(self, X, y):
        """Find optimal threshold that separates binary data."""
        thresholds = np.sort(X)
        max_f1 = 0
        optimal_C = thresholds[0]

        for C in thresholds:
            predictions = (X > C).astype(int)
            f1 = f1_score(y, predictions)

            if f1 > max_f1:
                max_f1 = f1
                optimal_C = C

        return optimal_C
    
    def _train_scaling_factor(self, X, y, threshold):
        """Find optimal scaling factor that maximizes ROC curve."""
        max_roc = 0
        optimal_K = None
        
        for k in np.linspace(0.0001, 1, 1000):
            probabilities = safe_sigmoid(k * (np.array(X) - threshold))
            
            try:
                roc = roc_auc_score(y, probabilities)
                if roc > max_roc:
                    max_roc = roc
                    optimal_K = k
            except ValueError:
                continue  # Skip if ROC cannot be computed
        
        return optimal_K if optimal_K is not None else 0.1
    
    def evaluate(self, dataset: Dataset) -> dict:
        """Evaluate the model on a dataset."""
        texts = [sample.output for sample in dataset]
        labels = [1 if sample.label == 'human' else 0 for sample in dataset]
        
        perplexities = self._compute_perplexities(texts)
        predictions = safe_sigmoid(
            self.model.scaling_factor * (np.array(perplexities) - self.model.perplexity_threshold)
        )
        
        conclusion = calculate_classification(labels, predictions)
        return conclusion.__dict__
    
    def save_model(self, path: str):
        """Save the trained model."""
        Path(path).mkdir(parents=True, exist_ok=True)
        weights = self.model.store_weights()
        
        with open(Path(path) / 'model.pkl', 'wb') as f:
            f.write(weights)
    
    def load_model(self, path: str):
        """Load a trained model."""
        with open(Path(path) / 'model.pkl', 'rb') as f:
            weights = f.read()
        
        self.model.load_weights(weights) 