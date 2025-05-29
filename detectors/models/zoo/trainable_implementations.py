"""Trainable implementations of detector models.

This module provides trainable versions of detector models that implement
both the IDetector and ITrainable interfaces.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
import pickle
import io

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoConfig
)

from detectors.models.zoo.implementations import HuggingFaceDetector
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.training.interfaces import ITrainable
from detectors.perplexity.model import PerplexityModel
from detectors.ghostbuster.model import GhostbusterDetector
from detectors.data.datasets import Dataset

logger = logging.getLogger(__name__)


class TrainableHuggingFaceDetector(HuggingFaceDetector, ITrainable):
    """Trainable version of HuggingFace detector.
    
    This class extends HuggingFaceDetector with training capabilities,
    implementing the ITrainable interface while maintaining backward
    compatibility with the IDetector interface.
    """
    
    def __init__(self, config: HuggingFaceConfig, freeze_base_model: bool = False):
        """Initialize trainable detector.
        
        Args:
            config: HuggingFace model configuration.
            freeze_base_model: Whether to freeze the base model.
        """
        super().__init__(config)
        self._freeze_base_model = freeze_base_model
        self._is_training = False
        
        if self._freeze_base_model:
            self._freeze_base_parameters()
    
    def _freeze_base_parameters(self):
        """Freeze base model parameters, only train classification head."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'pooler' not in name:
                param.requires_grad = False
    
    def train_mode(self, training: bool = True):
        """Set the model to training or evaluation mode."""
        self._is_training = training
        self.model.train(training)
    
    def is_training(self) -> bool:
        """Check if model is in training mode."""
        return self._is_training
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Get trainable parameters."""
        return {name: param for name, param in self.model.named_parameters() 
                if param.requires_grad}
    
    def compute_loss(self, texts: List[str], labels: List[str]) -> torch.Tensor:
        """Compute training loss."""
        if not self._is_training:
            raise RuntimeError("Model must be in training mode to compute loss")
        
        # Create label mapping
        unique_labels = list(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert labels to tensor
        label_ids = torch.tensor([label_to_id[label] for label in labels], 
                                dtype=torch.long)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Forward pass
        outputs = self.model(**inputs, labels=label_ids)
        return outputs.loss
    
    def update_parameters(self, **kwargs):
        """Update model parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'freeze_base_model': self._freeze_base_model,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'freeze_base_model' in checkpoint:
            self._freeze_base_model = checkpoint['freeze_base_model']
            if self._freeze_base_model:
                self._freeze_base_parameters()
        return checkpoint.get('metadata', {})
    
    def prepare_batch_inputs(self, texts: List[str], labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Prepare a batch of texts for training.
        
        Args:
            texts: List of input texts.
            labels: Optional list of label indices.
            
        Returns:
            Dictionary of model inputs.
        """
        # Tokenize texts
        inputs = self._prepare_inputs(texts)
        
        # Add labels if provided
        if labels is not None:
            inputs['labels'] = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return inputs
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[HuggingFaceConfig] = None) -> 'TrainableHuggingFaceDetector':
        """Load a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model.
            config: Optional configuration (will be loaded from model_path if not provided).
            
        Returns:
            TrainableHuggingFaceDetector instance.
        """
        if config is None:
            # Try to load config from checkpoint
            config_path = Path(model_path) / "detector_config.json"
            if config_path.exists():
                config = HuggingFaceConfig.load(str(config_path))
            else:
                # Create config from model
                model_config = AutoConfig.from_pretrained(model_path)
                config = HuggingFaceConfig(
                    model_name=model_path,
                    num_labels=model_config.num_labels
                )
        
        # Create instance
        detector = cls(config)
        
        # Load checkpoint
        detector.load_checkpoint(model_path)
        
        return detector
    
    def freeze_base_model(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the base model parameters.
        
        Useful for fine-tuning only the classification head.
        
        Args:
            freeze: Whether to freeze the base model.
        """
        # Freeze/unfreeze all parameters except classifier
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # Keep classifier trainable
                param.requires_grad = not freeze
        
        logger.info(f"{'Froze' if freeze else 'Unfroze'} base model parameters")
    
    def get_num_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def store_weights(self) -> bytes:
        """Store model weights as bytes."""
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()
    
    def load_weights(self, weights: bytes) -> None:
        """Load model weights from bytes."""
        buffer = io.BytesIO(weights)
        state_dict = torch.load(buffer, map_location='cpu')
        self.model.load_state_dict(state_dict)


class TrainablePerplexityModel(PerplexityModel, ITrainable):
    """Trainable version of PerplexityModel."""
    
    def __init__(self, model_handle: str = None, perplexity_threshold: float = None, 
                 scaling_factor: float = None):
        super().__init__(model_handle, perplexity_threshold, scaling_factor)
        self._is_training = False
    
    def train_mode(self, training: bool = True):
        """Set training mode."""
        self._is_training = training
    
    def is_training(self) -> bool:
        """Check if in training mode."""
        return self._is_training
    
    def get_num_parameters(self) -> int:
        """Get number of parameters (threshold and scaling factor)."""
        return 2  # perplexity_threshold and scaling_factor
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return 2  # Only threshold and scaling factor are trainable
    
    def get_trainable_parameters(self) -> Dict[str, Any]:
        """Get trainable parameters."""
        return {
            'perplexity_threshold': self.perplexity_threshold,
            'scaling_factor': self.scaling_factor
        }
    
    def compute_loss(self, texts: List[str], labels: List[str]) -> float:
        """Compute training loss (for optimization purposes)."""
        # This would be used in the context of finding optimal threshold/scaling
        # In practice, this is handled by the training logic
        return 0.0
    
    def update_parameters(self, **kwargs):
        """Update model parameters."""
        if 'perplexity_threshold' in kwargs:
            self.perplexity_threshold = kwargs['perplexity_threshold']
        if 'scaling_factor' in kwargs:
            self.scaling_factor = kwargs['scaling_factor']
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            'perplexity_threshold': self.perplexity_threshold,
            'scaling_factor': self.scaling_factor,
            'model_handle': self.model_handle,
            'metadata': metadata or {}
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.perplexity_threshold = checkpoint['perplexity_threshold']
        self.scaling_factor = checkpoint['scaling_factor']
        self.model_handle = checkpoint['model_handle']
        self.set_model(self.model_handle)
        
        return checkpoint.get('metadata', {})
    
    def store_weights(self) -> bytes:
        """Store model weights as bytes."""
        weights = {
            'perplexity_threshold': self.perplexity_threshold,
            'scaling_factor': self.scaling_factor,
            'model_handle': self.model_handle
        }
        buffer = io.BytesIO()
        pickle.dump(weights, buffer)
        return buffer.getvalue()
    
    def load_weights(self, weights: bytes) -> None:
        """Load model weights from bytes."""
        buffer = io.BytesIO(weights)
        weights_dict = pickle.load(buffer)
        self.perplexity_threshold = weights_dict['perplexity_threshold']
        self.scaling_factor = weights_dict['scaling_factor']
        self.model_handle = weights_dict['model_handle']
        self.set_model(self.model_handle)


class TrainableGhostbusterDetector(GhostbusterDetector, ITrainable):
    """Trainable version of GhostbusterDetector."""
    
    def __init__(self, clf=None, estimators=None):
        super().__init__(clf, estimators)
        self._is_training = False
    
    def train_mode(self, training: bool = True):
        """Set training mode."""
        self._is_training = training
    
    def is_training(self) -> bool:
        """Check if in training mode."""
        return self._is_training
    
    def get_num_parameters(self) -> int:
        """Get number of parameters in the classifier."""
        if self.clf is None:
            return 0
        
        # For sklearn models, count the number of parameters
        total_params = 0
        for step_name, step in self.clf.named_steps.items():
            if hasattr(step, 'coef_'):
                total_params += step.coef_.size
            if hasattr(step, 'intercept_'):
                total_params += step.intercept_.size
        return total_params
    
    def get_num_trainable_parameters(self) -> int:
        """All parameters are trainable in sklearn models."""
        return self.get_num_parameters()
    
    def get_trainable_parameters(self) -> Dict[str, Any]:
        """Get trainable parameters."""
        if self.clf is None:
            return {}
        
        params = {}
        for step_name, step in self.clf.named_steps.items():
            if hasattr(step, 'coef_'):
                params[f'{step_name}_coef'] = step.coef_
            if hasattr(step, 'intercept_'):
                params[f'{step_name}_intercept'] = step.intercept_
        return params
    
    def compute_loss(self, texts: List[str], labels: List[str]) -> float:
        """Compute training loss."""
        # This would involve feature extraction and loss computation
        # In practice, this is handled by sklearn's training process
        return 0.0
    
    def update_parameters(self, **kwargs):
        """Update model parameters."""
        # For sklearn models, parameters are updated during fit()
        pass
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            'clf': self.clf,
            'estimators': self.estimators,
            'metadata': metadata or {}
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.clf = checkpoint['clf']
        self.estimators = checkpoint['estimators']
        
        return checkpoint.get('metadata', {})
    
    def store_weights(self) -> bytes:
        """Store model weights as bytes."""
        weights = {
            'clf': self.clf,
            'estimators': self.estimators
        }
        buffer = io.BytesIO()
        pickle.dump(weights, buffer)
        return buffer.getvalue()
    
    def load_weights(self, weights: bytes) -> None:
        """Load model weights from bytes."""
        buffer = io.BytesIO(weights)
        weights_dict = pickle.load(buffer)
        self.clf = weights_dict['clf']
        self.estimators = weights_dict['estimators'] 