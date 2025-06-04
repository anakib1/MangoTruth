"""HuggingFace Trainer implementation.

This module provides a trainer implementation that leverages the HuggingFace
transformers.Trainer class for training sequence classification models.
"""

from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path
import logging
import io
import numpy as np

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint

from detectors.data.datasets.interfaces import TextDatasetInterface, TextSample
from detectors.training.interfaces import ITrainer, ITrainable, TrainingResult, Conclusion
from detectors.training.configs import HuggingFaceTrainingConfig
from detectors.models.zoo.evaluation import ModelEvaluator
from detectors.metrics import SplitConclusion

logger = logging.getLogger(__name__)


class HuggingFaceDataset(TorchDataset):
    """PyTorch Dataset wrapper for TextDatasetInterface."""

    def __init__(self,
                 dataset: TextDatasetInterface,
                 tokenizer_fn: Callable[[List[str]], Dict[str, Any]],
                 label_mapping: Dict[str, int]):
        """Initialize the dataset wrapper.
        
        Args:
            dataset: TextDatasetInterface instance.
            tokenizer_fn: Function to tokenize texts.
            label_mapping: Mapping from label strings to indices.
        """
        self.dataset = dataset
        self.tokenizer_fn = tokenizer_fn
        self.label_mapping = label_mapping

        # Pre-compute all samples for efficiency
        self.samples = list(self.dataset)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item."""
        sample = self.samples[idx]

        # Tokenize the text
        inputs = self.tokenizer_fn([sample.output])

        # Convert to single item (remove batch dimension)
        item = {k: v[0] for k, v in inputs.items()}

        # Add label
        item['labels'] = self.label_mapping.get(sample.label, 0)

        return item


class LoggingCallback(TrainerCallback):
    """Custom callback for additional logging during training."""

    def __init__(self, trainer_instance: 'HuggingFaceTrainer'):
        self.trainer_instance = trainer_instance

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics."""
        if logs:
            # Store metrics in training history
            for key, value in logs.items():
                if key not in self.trainer_instance._training_history:
                    self.trainer_instance._training_history[key] = []
                self.trainer_instance._training_history[key].append(value)


class HuggingFaceTrainer(ITrainer):
    """Trainer implementation using HuggingFace transformers.Trainer."""

    def __init__(self, model: ITrainable, config: HuggingFaceTrainingConfig):
        """Initialize the trainer.
        
        Args:
            model: Trainable model instance.
            config: Training configuration.
        """
        self.model = model
        self.config = config
        self._training_history = {}
        self._evaluator = ModelEvaluator(model)

    def _create_label_mapping(self, dataset: TextDatasetInterface) -> Dict[str, int]:
        """Create mapping from dataset labels to model label indices.
        
        Args:
            dataset: Dataset to extract labels from.
            
        Returns:
            Dictionary mapping label strings to indices.
        """
        dataset_labels = dataset.get_labels()
        model_labels = self.model.get_labels()

        label_mapping = {}
        for i, model_label in enumerate(model_labels):
            label_mapping[model_label] = i
            # Also map lowercase versions
            label_mapping[model_label.lower()] = i

        # Map dataset labels that might not exactly match
        for dataset_label in dataset_labels:
            if dataset_label not in label_mapping:
                # Try a case-insensitive match
                if dataset_label.lower() in label_mapping:
                    label_mapping[dataset_label] = label_mapping[dataset_label.lower()]
                else:
                    # Default to first label
                    logger.warning(f"Unknown label '{dataset_label}', mapping to index 0")
                    label_mapping[dataset_label] = 0

        return label_mapping

    def _prepare_dataset(self, dataset: TextDatasetInterface) -> HuggingFaceDataset:
        """Prepare dataset for training.
        
        Args:
            dataset: TextDatasetInterface instance.
            
        Returns:
            HuggingFaceDataset instance.
        """
        label_mapping = self._create_label_mapping(dataset)

        # Create tokenizer function
        def tokenizer_fn(texts: List[str]) -> Dict[str, Any]:
            return self.model.tokenizer(
                texts,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

        return HuggingFaceDataset(dataset, tokenizer_fn, label_mapping)

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
        start_time = datetime.now()

        # Set model to training mode
        self.model.train_mode(True)

        # Prepare datasets
        train_torch_dataset = self._prepare_dataset(train_dataset)
        eval_torch_dataset = None
        if validation_dataset:
            eval_torch_dataset = self._prepare_dataset(validation_dataset)

        # Create training arguments
        training_args = TrainingArguments(**self.config.to_hf_training_args())

        # Create data collator
        def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """Custom data collator for handling tokenized features."""
            if not features:
                return {}

            # Extract labels and create copies of features without labels
            labels = [f['labels'] for f in features]
            features_no_labels = [{k: v for k, v in f.items() if k != 'labels'} for f in features]

            # Use the tokenizer's built-in padding functionality for text features
            batch = self.model.tokenizer.pad(
                features_no_labels,
                padding=True,
                return_tensors="pt"
            )

            # Add labels back to the batch
            batch['labels'] = torch.tensor(labels, dtype=torch.long)

            return batch

        # Create trainer
        callbacks = [LoggingCallback(self)]
        if self.config.early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            ))

        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_torch_dataset,
            eval_dataset=eval_torch_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=None  # We'll evaluate separately for consistency
        )

        # Check for existing checkpoint
        last_checkpoint = None
        if Path(self.config.output_dir).exists():
            last_checkpoint = get_last_checkpoint(self.config.output_dir)
            if last_checkpoint:
                logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

        # Train
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save final model
        trainer.save_model()
        trainer.save_state()

        # Set model back to eval mode
        self.model.train_mode(False)

        # Evaluate on all splits
        logger.info("Evaluating on training set...")
        train_conclusion, _, _ = self._evaluator.evaluate_dataset(train_dataset)

        validation_conclusion = None
        if validation_dataset:
            logger.info("Evaluating on validation set...")
            validation_conclusion, _, _ = self._evaluator.evaluate_dataset(validation_dataset)

        test_conclusion = None
        if test_dataset:
            logger.info("Evaluating on test set...")
            test_conclusion, _, _ = self._evaluator.evaluate_dataset(test_dataset)

        # Get model weights
        model_weights = self.model.store_weights()

        # Create training result
        end_time = datetime.now()

        # Create Conclusion objects for validation and test if they exist
        validation_conclusion_obj = None
        if validation_conclusion:
            validation_conclusion_obj = Conclusion(
                weights=model_weights,
                detector_handle=self.config.model_name,
                datasets=[validation_dataset.__class__.__name__],
                train_conclusion=validation_conclusion,
                validation_conclusion=None  # No nested validation for the validation conclusion
            )

        test_conclusion_obj = None
        if test_conclusion:
            test_conclusion_obj = Conclusion(
                weights=model_weights,
                detector_handle=self.config.model_name,
                datasets=[test_dataset.__class__.__name__],
                train_conclusion=test_conclusion,
                validation_conclusion=None  # No nested validation for the test conclusion
            )

        result = TrainingResult(
            run_id=self.config.run_id,
            model_name=self.config.model_name,
            trainer_type="HuggingFaceTrainer",
            start_time=start_time,
            end_time=end_time,
            training_duration_seconds=(end_time - start_time).total_seconds(),
            train_conclusion=Conclusion(
                weights=model_weights,
                detector_handle=self.config.model_name,
                datasets=[train_dataset.__class__.__name__],
                train_conclusion=train_conclusion,
                validation_conclusion=validation_conclusion
            ),
            validation_conclusion=validation_conclusion_obj,
            test_conclusion=test_conclusion_obj,
            model_weights=model_weights,
            best_checkpoint_path=trainer.state.best_model_checkpoint,
            training_history=self._training_history,
            config=self.config,
            metadata={
                'train_samples': len(train_dataset),
                'validation_samples': len(validation_dataset) if validation_dataset else 0,
                'test_samples': len(test_dataset) if test_dataset else 0,
                'final_train_loss': train_result.training_loss,
                'total_steps': trainer.state.global_step,
            }
        )

        return result

    def evaluate(self, dataset: TextDatasetInterface) -> Conclusion:
        """Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on.
            
        Returns:
            Conclusion with evaluation metrics.
        """
        self.model.train_mode(False)
        split_conclusion, _, _ = self._evaluator.evaluate_dataset(dataset)

        return Conclusion(
            weights=self.model.store_weights(),
            detector_handle=self.config.model_name,
            datasets=[dataset.__class__.__name__],
            train_conclusion=split_conclusion,
            validation_conclusion=split_conclusion
        )

    def save_model(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model.
        """
        self.model.save_checkpoint(path)

    def load_model(self, path: str) -> None:
        """Load a trained model.
        
        Args:
            path: Path to load the model from.
        """
        self.model.load_checkpoint(path)
