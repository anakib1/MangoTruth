"""Trainer for Ghostbuster models."""

import time
import numpy as np
from pathlib import Path
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from detectors.training.interfaces import ITrainer, TrainingResult
from detectors.training.configs import BaseTrainingConfig
from detectors.training.trainers.base_trainer import BaseTrainer
from detectors.models.zoo.trainable_implementations import TrainableGhostbusterDetector
from detectors.data.datasets import Dataset
from detectors.ghostbuster.features import extract_features
from detectors.ghostbuster.ngrams import UnigramModel, TrigramModel
from detectors.ghostbuster.openai import OpenaiProbabilityEstimator
from detectors.utils.training import calculate_classification


class GhostbusterTrainingConfig(BaseTrainingConfig):
    """Configuration for Ghostbuster model training."""
    
    def __init__(
        self,
        tokenizer_handle: str = 'gugarosa/cl100k_base',
        llm_handles: List[str] = None,
        max_length: int = 15000,
        validation_split: float = 0.3,
        classifier_config: dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer_handle = tokenizer_handle
        self.llm_handles = llm_handles or ['babbage-002']
        self.max_length = max_length
        self.validation_split = validation_split
        self.classifier_config = classifier_config or {
            'C': 1,
            'max_iter': 1000
        }
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        config_dict = super().to_dict()
        config_dict.update({
            'tokenizer_handle': self.tokenizer_handle,
            'llm_handles': self.llm_handles,
            'max_length': self.max_length,
            'classifier_config': self.classifier_config
        })
        return config_dict


class GhostbusterTrainer(BaseTrainer):
    """Trainer for Ghostbuster models."""
    
    def __init__(self, model: TrainableGhostbusterDetector, config: GhostbusterTrainingConfig):
        super().__init__(model, config)
        self.config: GhostbusterTrainingConfig = config
        self.language_models = None
    
    def train(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> TrainingResult:
        """Train the Ghostbuster model."""
        from datetime import datetime
        
        start_time = datetime.now()
        
        # Convert dataset to texts and labels
        train_texts = [sample.output for sample in train_dataset 
                      if len(sample.output) < self.config.max_length]
        train_labels = [1 if sample.label == 'ai' or sample.label == 'llm' else 0 
                       for sample in train_dataset 
                       if len(sample.output) < self.config.max_length]
        
        # Create language models
        self._setup_language_models(train_texts)
        
        # Extract features
        features = self._extract_features(train_texts)
        
        # Split data if validation not provided
        if validation_dataset is None:
            X_train, X_val, y_train, y_val = train_test_split(
                features, train_labels,
                test_size=self.config.validation_split,
                stratify=train_labels,
                random_state=42
            )
        else:
            X_train, y_train = features, train_labels
            val_texts = [sample.output for sample in validation_dataset
                        if len(sample.output) < self.config.max_length]
            val_labels = [1 if sample.label == 'ai' or sample.label == 'llm' else 0 
                         for sample in validation_dataset
                         if len(sample.output) < self.config.max_length]
            X_val = self._extract_features(val_texts)
            y_val = val_labels
        
        # Train classifier
        classifier = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LogisticRegression(**self.config.classifier_config)
            )
        )
        
        classifier.fit(X_train, y_train)
        
        # Update model
        self.model.clf = classifier
        self.model.estimators = self.language_models
        
        # Get predictions
        y_train_pred = classifier.predict_proba(X_train)[:, 1]
        y_val_pred = classifier.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_conclusion = calculate_classification(y_train, y_train_pred)
        val_conclusion = calculate_classification(y_val, y_val_pred)
        
        # Handle test set if provided
        test_conclusion = None
        if test_dataset is not None:
            test_texts = [sample.output for sample in test_dataset
                         if len(sample.output) < self.config.max_length]
            test_labels = [1 if sample.label == 'ai' or sample.label == 'llm' else 0 
                          for sample in test_dataset
                          if len(sample.output) < self.config.max_length]
            X_test = self._extract_features(test_texts)
            y_test_pred = classifier.predict_proba(X_test)[:, 1]
            test_conclusion = calculate_classification(test_labels, y_test_pred)
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        return TrainingResult(
            run_id=self.config.run_id,
            model_name="ghostbuster",
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
                'tokenizer_handle': self.config.tokenizer_handle,
                'llm_handles': self.config.llm_handles,
                'classifier_config': self.config.classifier_config,
                'total_samples': len(train_texts)
            }
        )
    
    def _setup_language_models(self, train_texts: List[str]):
        """Setup and train language models."""
        from tqdm.auto import tqdm
        
        # Create n-gram models
        unigram = UnigramModel(tokenizer_handle=self.config.tokenizer_handle)
        unigram.train(train_texts)
        
        trigram = TrigramModel(tokenizer_handle=self.config.tokenizer_handle)
        trigram.train(train_texts)
        
        # Create OpenAI probability estimator
        estimator = OpenaiProbabilityEstimator(model_name=self.config.llm_handles[0])
        
        # Verify tokenizer compatibility
        print("Verifying tokenizer compatibility...")
        for _ in range(min(10, len(train_texts))):
            i = np.random.randint(0, len(train_texts))
            text = train_texts[i]
            try:
                est_tokens = estimator.get_text_log_proba(text)[1]
                uni_tokens = unigram.get_text_log_proba(text)[1]
                assert est_tokens.shape == uni_tokens.shape, f"Token shape mismatch: {est_tokens.shape} vs {uni_tokens.shape}"
            except Exception as e:
                print(f"Warning: Tokenizer compatibility issue: {e}")
                break
        
        self.language_models = [unigram, trigram, estimator]
    
    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from texts."""
        from tqdm.auto import tqdm
        
        features = []
        for text in tqdm(texts, desc="Extracting features"):
            log_probas = [model.get_text_log_proba(text)[1] for model in self.language_models]
            feature_vector = extract_features(log_probas)
            features.append(feature_vector)
        
        return np.array(features)
    
    def evaluate(self, dataset: Dataset) -> dict:
        """Evaluate the model on a dataset."""
        texts = [sample.output for sample in dataset
                if len(sample.output) < self.config.max_length]
        labels = [1 if sample.label == 'ai' or sample.label == 'llm' else 0 
                 for sample in dataset
                 if len(sample.output) < self.config.max_length]
        
        features = self._extract_features(texts)
        predictions = self.model.clf.predict_proba(features)[:, 1]
        
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