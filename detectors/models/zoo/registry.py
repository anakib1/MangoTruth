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
    """Enhanced registry for managing available models with optional Nexus integration.
    
    This registry supports:
    - Local model registration and instantiation
    - Loading pre-trained models from any Nexus implementation
    - Default model presets for easy usage
    - Model discovery and recommendation
    
    The registry works with any Nexus implementation (Neptune, HuggingFace, etc.)
    and provides fallback mechanisms when specific features aren't available.
    """
    
    def __init__(self, nexus: Optional[Nexus] = None):
        """Initialize the model registry.
        
        Args:
            nexus: Optional Nexus instance for loading pre-trained models from remote storage.
                   Can be any implementation (NeptuneNexus, HuggingFaceNexus, etc.)
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
        
        This method works with any Nexus implementation by using only the core
        Nexus interface methods (load_run_weights, store_run_weights).
        
        Args:
            model_name: Name of the pre-trained model, run ID, or model identifier
            **kwargs: Additional configuration parameters
            
        Returns:
            Loaded pre-trained model
            
        Raises:
            ValueError: If model not found or nexus not configured
        """
        # First, check if it's a known pre-trained model with metadata
        model_info = get_model_info(model_name)
        if model_info:
            return self._load_from_pretrained_info(model_info, **kwargs)
        
        # Check if it's a UUID (run_id)
        try:
            run_id = UUID(model_name)
            return self._load_from_run_id(run_id, **kwargs)
        except ValueError:
            pass
        
        # Check if it looks like a repo ID and we have a HuggingFace-capable nexus
        if "/" in model_name and self._is_huggingface_nexus():
            return self._load_from_hub_repo(model_name, **kwargs)
        
        # Fall back to regular model loading
        if model_name in self._models:
            return self.get_model(model_name, **kwargs)
            
        raise ValueError(
            f"Model '{model_name}' not found. Available options:\n"
            f"- Pre-trained models: {list(PRETRAINED_MODELS.keys())}\n"
            f"- Registered models: {list(self._models.keys())}\n"
            f"- Run UUIDs (if nexus available): {bool(self.nexus)}"
        )

    def _load_from_pretrained_info(self, model_info: PretrainedModelInfo, **kwargs) -> IDetector:
        """Load model from pre-trained model information."""
        # Get the model class
        model_class = self._get_model_class(model_info.model_class)
        
        # Update config with any provided kwargs
        config_dict = model_info.config.to_dict()
        config_dict.update(kwargs)
        
        # Create config
        config = self._create_config_instance(model_info.config, config_dict)
        
        # Create model instance
        model = model_class(config)
        
        # Load weights if available and we have a nexus
        if model_info.run_id and self.nexus:
            try:
                run_id = UUID(model_info.run_id)
                weights = self.nexus.load_run_weights(run_id)
                model.load_weights(weights)
                logger.info(f"Loaded pre-trained weights for {model_info.name}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained weights for {model_info.name}: {e}")
        elif model_info.run_id and not self.nexus:
            logger.warning(f"No nexus available to load weights for {model_info.name}")
        
        return model

    def _load_from_run_id(self, run_id: UUID, **kwargs) -> IDetector:
        """Load model directly from a run ID using any Nexus implementation."""
        if not self.nexus:
            raise ValueError("Nexus is required to load models from run IDs")
        
        try:
            # Try to get metadata if the nexus supports it
            metadata = self._get_run_metadata(run_id)
            
            if metadata:
                # Use metadata to determine model type
                detector_handle = metadata.get('detector_handle', 'huggingface')
                model_class = self._determine_model_class_from_handle(detector_handle)
                
                # Create default config and update with kwargs
                config = self._create_default_config_for_class(model_class, **kwargs)
            else:
                # Fallback: assume HuggingFace model
                logger.warning(f"No metadata available for run {run_id}, assuming HuggingFace model")
                from detectors.models.zoo.implementations import HuggingFaceDetector
                model_class = HuggingFaceDetector
                config = HuggingFaceConfig(
                    model_name="bert-base-uncased",  # Default, will be overridden by weights
                    **kwargs
                )
            
            # Create model and load weights
            model = model_class(config)
            weights = self.nexus.load_run_weights(run_id)
            model.load_weights(weights)
            
            logger.info(f"Successfully loaded model from run {run_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from run {run_id}: {e}")
            raise ValueError(f"Failed to load model from run {run_id}: {e}")

    def _load_from_hub_repo(self, repo_id: str, **kwargs) -> IDetector:
        """Load model from HuggingFace Hub repository (only if HuggingFace nexus is available)."""
        if not self._is_huggingface_nexus():
            raise ValueError("HuggingFace Hub loading requires HuggingFaceNexus")
        
        try:
            # Extract run_id from repo_id (HuggingFace-specific format)
            if "plagiarism-detector-" in repo_id:
                run_id_str = repo_id.split("plagiarism-detector-")[-1]
                run_id = UUID(run_id_str)
                return self._load_from_run_id(run_id, **kwargs)
            else:
                raise ValueError(f"Cannot parse run_id from repo_id: {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model from HuggingFace Hub {repo_id}: {e}")
            raise ValueError(f"Failed to load model from HuggingFace Hub: {e}")

    def _get_model_class(self, model_class_name: str) -> Type[IDetector]:
        """Get model class by name."""
        if model_class_name == "HuggingFaceDetector":
            from detectors.models.zoo.implementations import HuggingFaceDetector
            return HuggingFaceDetector
        elif model_class_name == "PerplexityModel":
            from detectors.perplexity.model import PerplexityModel
            return PerplexityModel
        elif model_class_name == "GhostbusterDetector":
            from detectors.ghostbuster.model import GhostbusterDetector
            return GhostbusterDetector
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

    def _create_config_instance(self, original_config: ModelConfig, config_dict: Dict) -> ModelConfig:
        """Create a config instance of the appropriate type."""
        if isinstance(original_config, HuggingFaceConfig):
            return HuggingFaceConfig(**config_dict)
        else:
            return type(original_config)(**config_dict)

    def _is_huggingface_nexus(self) -> bool:
        """Check if the current nexus is a HuggingFace nexus."""
        return (self.nexus and 
                hasattr(self.nexus, 'get_model_metadata') and 
                hasattr(self.nexus, 'list_available_models'))

    def _get_run_metadata(self, run_id: UUID) -> Optional[Dict]:
        """Get metadata for a run if the nexus supports it."""
        if hasattr(self.nexus, 'get_model_metadata'):
            try:
                return self.nexus.get_model_metadata(run_id)
            except Exception as e:
                logger.debug(f"Failed to get metadata for run {run_id}: {e}")
        return None

    def _determine_model_class_from_handle(self, detector_handle: str) -> Type[IDetector]:
        """Determine model class from detector handle."""
        handle_lower = detector_handle.lower()
        
        if 'huggingface' in handle_lower or 'bert' in handle_lower or 'roberta' in handle_lower:
            from detectors.models.zoo.implementations import HuggingFaceDetector
            return HuggingFaceDetector
        elif 'perplexity' in handle_lower or 'gpt' in handle_lower:
            from detectors.perplexity.model import PerplexityModel
            return PerplexityModel
        elif 'ghostbuster' in handle_lower:
            from detectors.ghostbuster.model import GhostbusterDetector
            return GhostbusterDetector
        else:
            # Default to HuggingFace
            from detectors.models.zoo.implementations import HuggingFaceDetector
            return HuggingFaceDetector

    def _create_default_config_for_class(self, model_class: Type[IDetector], **kwargs) -> ModelConfig:
        """Create a default config for a model class."""
        if 'HuggingFace' in model_class.__name__:
            return HuggingFaceConfig(
                model_name="bert-base-uncased",
                **kwargs
            )
        else:
            # For other model types, we'd need their specific config classes
            # For now, return a basic config
            from detectors.models.zoo.configs import ModelConfig
            return ModelConfig(
                model_name="unknown",
                model_type="unknown",
                **kwargs
            )

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
                "repo_id": model_info.repo_id,
                "has_weights": bool(model_info.run_id and self.nexus)
            }
        
        # Add available models from nexus if it supports listing
        if self._is_huggingface_nexus():
            try:
                hub_models = self.nexus.list_available_models()
                for model_info in hub_models:
                    model_id = model_info.get("model_id", "unknown")
                    models[f"hub_{model_id}"] = {
                        "type": "hub",
                        "model_id": model_id,
                        "downloads": model_info.get("downloads", 0),
                        "last_modified": model_info.get("last_modified"),
                        "tags": model_info.get("tags", [])
                    }
            except Exception as e:
                logger.debug(f"Failed to list hub models: {e}")
        
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
            nexus: Nexus instance for remote model storage (any implementation)
        """
        self.nexus = nexus
        nexus_type = type(nexus).__name__
        logger.info(f"Updated Nexus instance for model registry: {nexus_type}")

    def discover_models(self) -> Dict[str, List[str]]:
        """Discover available models by category.
        
        Returns:
            Dictionary mapping categories to model lists
        """
        discovery = {
            "registered": list(self._models.keys()),
            "pretrained": list(PRETRAINED_MODELS.keys()),
            "categories": {},
            "use_cases": {},
            "nexus_type": type(self.nexus).__name__ if self.nexus else None,
            "nexus_capabilities": self._get_nexus_capabilities()
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

    def _get_nexus_capabilities(self) -> Dict[str, bool]:
        """Get capabilities of the current nexus."""
        if not self.nexus:
            return {"available": False}
        
        return {
            "available": True,
            "load_weights": hasattr(self.nexus, 'load_run_weights'),
            "store_weights": hasattr(self.nexus, 'store_run_weights'),
            "metadata": hasattr(self.nexus, 'get_model_metadata'),
            "list_models": hasattr(self.nexus, 'list_available_models'),
            "huggingface_features": self._is_huggingface_nexus(),
            "training_support": hasattr(self.nexus, 'conclude_run')
        } 