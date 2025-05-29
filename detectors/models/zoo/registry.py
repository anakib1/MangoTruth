from typing import Dict, Type, Optional, Any, List
from uuid import UUID, uuid4
import logging

from detectors.models.zoo.configs import ModelConfig, HuggingFaceConfig
from detectors.interfaces import IDetector, Nexus
from detectors.models.zoo.default_models import (
    PRETRAINED_MODELS, 
    DEFAULT_HUGGINGFACE_CONFIGS,
    PretrainedModelInfo,
    get_model_info,
    list_available_models,
    get_models_by_category,
    get_recommended_models
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Enhanced registry for managing available models with Nexus integration.
    
    This registry supports:
    - Local model registration and instantiation
    - Loading pre-trained models from HuggingFace Hub via Nexus
    - Default model presets for easy usage
    - Model discovery and recommendation
    """
    
    def __init__(self, nexus: Optional[Nexus] = None):
        """Initialize the model registry.
        
        Args:
            nexus: Nexus instance for loading pre-trained models from remote storage
        """
        self.nexus = nexus
        self._models: Dict[str, Type[IDetector]] = {}
        self._configs: Dict[str, ModelConfig] = {}
        
        # Initialize default configurations
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize default model configurations."""
        for name, config in DEFAULT_HUGGINGFACE_CONFIGS.items():
            self._configs[name] = config

    def register_model(self, name: str, model_class: Type[IDetector], config: Optional[ModelConfig] = None):
        """Register a new model with its configuration.
        
        Args:
            name: Unique name for the model
            model_class: Model class implementing IDetector
            config: Optional model configuration
        """
        if name in self._models:
            logger.warning(f"Model {name} is already registered. Overwriting.")
        
        self._models[name] = model_class
        if config:
            self._configs[name] = config
        elif name in DEFAULT_HUGGINGFACE_CONFIGS:
            self._configs[name] = DEFAULT_HUGGINGFACE_CONFIGS[name]

    def get_model(self, name: str, **kwargs) -> IDetector:
        """Get a model instance with specified configuration.
        
        Args:
            name: Name of the model to load
            **kwargs: Additional configuration parameters
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model {name} is not registered. Available models: {list(self._models.keys())}")
        
        model_class = self._models[name]
        config = self._configs.get(name)
        
        if config:
            # Create a copy and update with kwargs
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            
            # Recreate config with updated parameters
            if isinstance(config, HuggingFaceConfig):
                updated_config = HuggingFaceConfig(**config_dict)
            else:
                updated_config = type(config)(**config_dict)
            
            return model_class(updated_config)
        
        return model_class(**kwargs)

    def load_pretrained_model(self, model_name: str, **kwargs) -> IDetector:
        """Load a pre-trained model from the registry or remote storage.
        
        Args:
            model_name: Name of the pre-trained model or HuggingFace repo ID
            **kwargs: Additional configuration parameters
            
        Returns:
            Loaded pre-trained model
            
        Raises:
            ValueError: If model not found or nexus not configured
        """
        # Check if it's a known pre-trained model
        model_info = get_model_info(model_name)
        if model_info:
            return self._load_from_pretrained_info(model_info, **kwargs)
        
        # Try to load from HuggingFace Hub directly
        if self.nexus and "/" in model_name:
            return self._load_from_hub_repo(model_name, **kwargs)
        
        # Fall back to regular model loading
        if model_name in self._models:
            return self.get_model(model_name, **kwargs)
            
        raise ValueError(f"Pre-trained model '{model_name}' not found. "
                        f"Available models: {list(PRETRAINED_MODELS.keys())}")

    def _load_from_pretrained_info(self, model_info: PretrainedModelInfo, **kwargs) -> IDetector:
        """Load model from pre-trained model information."""
        if not self.nexus:
            raise ValueError("Nexus is required to load pre-trained models from remote storage")
        
        # Get the model class
        if model_info.model_class == "HuggingFaceDetector":
            from detectors.models.zoo.implementations import HuggingFaceDetector
            model_class = HuggingFaceDetector
        elif model_info.model_class == "PerplexityModel":
            from detectors.perplexity.model import PerplexityModel
            model_class = PerplexityModel
        elif model_info.model_class == "GhostbusterDetector":
            from detectors.ghostbuster.model import GhostbusterDetector
            model_class = GhostbusterDetector
        else:
            raise ValueError(f"Unknown model class: {model_info.model_class}")
        
        # Update config with any provided kwargs
        config_dict = model_info.config.to_dict()
        config_dict.update(kwargs)
        
        # Create config
        if isinstance(model_info.config, HuggingFaceConfig):
            config = HuggingFaceConfig(**config_dict)
        else:
            config = type(model_info.config)(**config_dict)
        
        # Create model instance
        model = model_class(config)
        
        # Load weights if available
        if model_info.run_id:
            try:
                run_id = UUID(model_info.run_id)
                weights = self.nexus.load_run_weights(run_id)
                model.load_weights(weights)
                logger.info(f"Loaded pre-trained weights for {model_info.name}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained weights for {model_info.name}: {e}")
        
        return model

    def _load_from_hub_repo(self, repo_id: str, **kwargs) -> IDetector:
        """Load model directly from HuggingFace Hub repository."""
        if not self.nexus:
            raise ValueError("Nexus is required to load models from HuggingFace Hub")
        
        try:
            # Extract run_id from repo_id (assuming format: namespace/plagiarism-detector-{run_id})
            if "plagiarism-detector-" in repo_id:
                run_id_str = repo_id.split("plagiarism-detector-")[-1]
                run_id = UUID(run_id_str)
                
                # Try to get metadata first
                if hasattr(self.nexus, 'get_model_metadata'):
                    metadata = self.nexus.get_model_metadata(run_id)
                    if metadata:
                        # Create model based on metadata
                        detector_handle = metadata.get('detector_handle', 'huggingface')
                        
                        if 'huggingface' in detector_handle.lower():
                            from detectors.models.zoo.implementations import HuggingFaceDetector
                            config = HuggingFaceConfig(
                                model_name="bert-base-uncased",  # Default, will be overridden by weights
                                **kwargs
                            )
                            model = HuggingFaceDetector(config)
                        else:
                            raise ValueError(f"Unsupported detector handle: {detector_handle}")
                        
                        # Load weights
                        weights = self.nexus.load_run_weights(run_id)
                        model.load_weights(weights)
                        
                        return model
            
            raise ValueError(f"Could not parse run_id from repo_id: {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {repo_id}: {e}")
            raise ValueError(f"Failed to load model from HuggingFace Hub: {e}")

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations.
        
        Returns:
            Dictionary mapping model names to their information
        """
        models = {}
        
        # Add registered models
        for name, model_class in self._models.items():
            models[name] = {
                "type": "registered",
                "model_class": model_class.__name__,
                "config": self._configs.get(name).to_dict() if name in self._configs else None
            }
        
        # Add pre-trained models
        for name, model_info in PRETRAINED_MODELS.items():
            models[name] = {
                "type": "pretrained",
                "description": model_info.description,
                "model_class": model_info.model_class,
                "tags": model_info.tags,
                "performance": model_info.performance,
                "repo_id": model_info.repo_id
            }
        
        return models

    def list_pretrained_models(self) -> Dict[str, PretrainedModelInfo]:
        """List all available pre-trained models.
        
        Returns:
            Dictionary of pre-trained model information
        """
        return list_available_models()

    def get_models_by_category(self, category: str) -> List[PretrainedModelInfo]:
        """Get pre-trained models by category.
        
        Args:
            category: Category name (e.g., 'fast', 'accurate', 'multilingual')
            
        Returns:
            List of models in the specified category
        """
        return get_models_by_category(category)

    def get_recommended_models(self, use_case: str) -> List[PretrainedModelInfo]:
        """Get recommended models for a specific use case.
        
        Args:
            use_case: Use case (e.g., 'research', 'production', 'real-time')
            
        Returns:
            List of recommended models
        """
        return get_recommended_models(use_case)

    def get_config(self, name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.
        
        Args:
            name: Model name
            
        Returns:
            Model configuration or None if not found
        """
        return self._configs.get(name)

    def update_config(self, name: str, config: ModelConfig):
        """Update configuration for a specific model.
        
        Args:
            name: Model name
            config: New configuration
            
        Raises:
            ValueError: If model is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model {name} is not registered")
        self._configs[name] = config

    def register_custom_model(self, 
                            name: str, 
                            model_class: Type[IDetector], 
                            config: ModelConfig,
                            description: str = "",
                            tags: List[str] = None):
        """Register a custom model with additional metadata.
        
        Args:
            name: Unique name for the model
            model_class: Model class implementing IDetector
            config: Model configuration
            description: Model description
            tags: List of tags for categorization
        """
        self.register_model(name, model_class, config)
        
        # Store additional metadata (could be extended to persist this)
        logger.info(f"Registered custom model '{name}': {description}")
        if tags:
            logger.info(f"Tags: {', '.join(tags)}")

    def set_nexus(self, nexus: Nexus):
        """Set or update the Nexus instance.
        
        Args:
            nexus: Nexus instance for remote model storage
        """
        self.nexus = nexus
        logger.info("Updated Nexus instance for model registry")

    def discover_models(self) -> Dict[str, List[str]]:
        """Discover available models by category.
        
        Returns:
            Dictionary mapping categories to model lists
        """
        discovery = {
            "registered": list(self._models.keys()),
            "pretrained": list(PRETRAINED_MODELS.keys()),
            "categories": {},
            "use_cases": {}
        }
        
        # Add category breakdown
        from detectors.models.zoo.default_models import MODEL_CATEGORIES, USAGE_RECOMMENDATIONS
        for category, info in MODEL_CATEGORIES.items():
            discovery["categories"][category] = {
                "description": info["description"],
                "models": info["models"]
            }
        
        # Add use case breakdown
        for use_case, info in USAGE_RECOMMENDATIONS.items():
            discovery["use_cases"][use_case] = {
                "description": info["description"],
                "recommended_models": info["recommended_models"],
                "considerations": info["considerations"]
            }
        
        return discovery 