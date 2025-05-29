"""Training configuration classes.

This module provides specific configuration classes for different training
approaches while maintaining a common interface.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os

from detectors.training.interfaces import TrainingConfig


@dataclass
class BaseTrainingConfig(TrainingConfig):
    """Base implementation of TrainingConfig with serialization support."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "validation_split": self.validation_split,
            "validation_strategy": self.validation_strategy,
            "validation_steps": self.validation_steps,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "run_id": str(self.run_id),
            "extra_params": self.extra_params
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseTrainingConfig':
        """Create configuration from dictionary."""
        # Handle UUID conversion
        if "run_id" in config_dict and isinstance(config_dict["run_id"], str):
            from uuid import UUID
            config_dict["run_id"] = UUID(config_dict["run_id"])
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class HuggingFaceTrainingConfig(BaseTrainingConfig):
    """Configuration specific to HuggingFace Trainer.
    
    This configuration extends the base configuration with parameters
    specific to the HuggingFace transformers.Trainer class.
    """
    # Model configuration
    model_name: str = "bert-base-uncased"
    tokenizer_name: Optional[str] = None
    num_labels: int = 2
    max_length: int = 512
    
    # Optimization
    optimizer_name: str = "adamw_torch"  # "adamw_torch", "adamw_hf", "sgd", "adafactor"
    scheduler_type: str = "linear"  # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Training efficiency
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    tf32: bool = None  # None means use PyTorch default
    
    # Data loading
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # Evaluation
    evaluation_strategy: str = "epoch"  # Alias for validation_strategy
    eval_steps: Optional[int] = None  # Alias for validation_steps
    eval_accumulation_steps: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    
    # Model initialization
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # HuggingFace specific
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    
    # Metrics
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    def __post_init__(self):
        """Post-initialization to set dependent values."""
        # Set tokenizer name to model name if not specified
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
        
        # Set per-device batch sizes if not specified
        if self.per_device_train_batch_size is None:
            self.per_device_train_batch_size = self.batch_size
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.batch_size
        
        # Sync evaluation settings
        if self.evaluation_strategy != self.validation_strategy:
            self.evaluation_strategy = self.validation_strategy
        if self.eval_steps is None and self.validation_steps is not None:
            self.eval_steps = self.validation_steps
    
    def to_hf_training_args(self) -> Dict[str, Any]:
        """Convert to HuggingFace TrainingArguments format."""
        args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "evaluation_strategy": self.evaluation_strategy,
            "load_best_model_at_end": self.early_stopping,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "optim": self.optimizer_name,
            "lr_scheduler_type": self.scheduler_type,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "seed": self.seed,
            "push_to_hub": self.push_to_hub,
        }
        
        # Add optional parameters only if they are not None
        if self.warmup_steps is not None:
            args_dict["warmup_steps"] = self.warmup_steps
        if self.warmup_ratio is not None:
            args_dict["warmup_ratio"] = self.warmup_ratio
        if self.save_steps is not None:
            args_dict["save_steps"] = self.save_steps
        if self.save_total_limit is not None:
            args_dict["save_total_limit"] = self.save_total_limit
        if self.eval_steps is not None:
            args_dict["eval_steps"] = self.eval_steps
        if self.eval_accumulation_steps is not None:
            args_dict["eval_accumulation_steps"] = self.eval_accumulation_steps
        if self.hub_model_id is not None:
            args_dict["hub_model_id"] = self.hub_model_id
        if self.hub_token is not None:
            args_dict["hub_token"] = self.hub_token
        
        # Add tf32 if specified
        if self.tf32 is not None:
            args_dict["tf32"] = self.tf32
        
        # Add any extra parameters
        args_dict.update(self.extra_params)
        
        return args_dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, including HuggingFace-specific fields."""
        # Get base configuration
        config_dict = super().to_dict()
        
        # Add HuggingFace-specific fields
        config_dict.update({
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "optimizer_name": self.optimizer_name,
            "scheduler_type": self.scheduler_type,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "tf32": self.tf32,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "eval_accumulation_steps": self.eval_accumulation_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "model_kwargs": self.model_kwargs,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_token": self.hub_token,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better
        })
        
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HuggingFaceTrainingConfig':
        """Create configuration from dictionary."""
        # Handle UUID conversion
        if "run_id" in config_dict and isinstance(config_dict["run_id"], str):
            from uuid import UUID
            config_dict["run_id"] = UUID(config_dict["run_id"])
        return cls(**config_dict)


@dataclass
class ClassicalMLTrainingConfig(BaseTrainingConfig):
    """Configuration for classical machine learning models.
    
    This configuration is designed for non-deep-learning approaches
    like SVM, Random Forest, Naive Bayes, etc.
    """
    # Model type
    model_type: str = "svm"  # "svm", "random_forest", "naive_bayes", "logistic_regression", etc.
    
    # Feature extraction
    feature_type: str = "tfidf"  # "tfidf", "bow", "custom"
    max_features: Optional[int] = 10000
    ngram_range: tuple = (1, 2)
    
    # Model-specific parameters (will vary based on model_type)
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-validation
    cv_folds: int = 5
    
    # Note: Many inherited parameters like learning_rate, epochs, etc.
    # may not apply to all classical ML models 