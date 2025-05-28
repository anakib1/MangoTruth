from typing import Dict, Type, Optional, Any
from detectors.models.zoo.configs import ModelConfig, HuggingFaceConfig
from detectors.interfaces import IDetector
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing available models and their configurations"""
    
    def __init__(self):
        self._models: Dict[str, Type[IDetector]] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._default_configs: Dict[str, Dict[str, Any]] = {
            "bert-base-uncased": {
                "model_name": "bert-base-uncased",
                "model_type": "huggingface",
                "num_labels": 2,
                "max_length": 512
            },
            "roberta-base": {
                "model_name": "roberta-base",
                "model_type": "huggingface",
                "num_labels": 2,
                "max_length": 512
            }
        }

    def register_model(self, name: str, model_class: Type[IDetector], config: Optional[ModelConfig] = None):
        """Register a new model with its configuration"""
        if name in self._models:
            logger.warning(f"Model {name} is already registered. Overwriting.")
        
        self._models[name] = model_class
        if config:
            self._configs[name] = config
        elif name in self._default_configs:
            self._configs[name] = HuggingFaceConfig(**self._default_configs[name])

    def get_model(self, name: str, **kwargs) -> IDetector:
        """Get a model instance with specified configuration"""
        if name not in self._models:
            raise ValueError(f"Model {name} is not registered")
        
        model_class = self._models[name]
        config = self._configs.get(name)
        
        if config:
            # Update config with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return model_class(config)
        
        return model_class(**kwargs)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations"""
        return {
            name: {
                "model_class": model_class.__name__,
                "config": self._configs.get(name).to_dict() if name in self._configs else None
            }
            for name, model_class in self._models.items()
        }

    def get_config(self, name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self._configs.get(name)

    def update_config(self, name: str, config: ModelConfig):
        """Update configuration for a specific model"""
        if name not in self._models:
            raise ValueError(f"Model {name} is not registered")
        self._configs[name] = config 