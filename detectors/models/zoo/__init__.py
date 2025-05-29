from detectors.models.zoo.registry import ModelRegistry
from detectors.models.zoo.configs import ModelConfig, HuggingFaceConfig
from detectors.models.zoo.implementations import HuggingFaceDetector
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.models.zoo.evaluation import evaluate, ModelEvaluator
from detectors.models.zoo.hub_nexus import HuggingFaceNexus
from detectors.models.zoo.default_models import (
    PretrainedModelInfo,
    PRETRAINED_MODELS,
    DEFAULT_HUGGINGFACE_CONFIGS,
    MODEL_CATEGORIES,
    USAGE_RECOMMENDATIONS,
    get_model_info,
    list_available_models,
    get_models_by_category,
    get_recommended_models,
    get_default_config
)

__all__ = [
    # Core components
    'ModelRegistry', 
    'ModelConfig', 
    'HuggingFaceConfig', 
    'HuggingFaceDetector',
    'TrainableHuggingFaceDetector',
    'evaluate',
    'ModelEvaluator',
    
    # HuggingFace Hub integration
    'HuggingFaceNexus',
    
    # Default models and presets
    'PretrainedModelInfo',
    'PRETRAINED_MODELS',
    'DEFAULT_HUGGINGFACE_CONFIGS',
    'MODEL_CATEGORIES',
    'USAGE_RECOMMENDATIONS',
    'get_model_info',
    'list_available_models',
    'get_models_by_category',
    'get_recommended_models',
    'get_default_config'
] 