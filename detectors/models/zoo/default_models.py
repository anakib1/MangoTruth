"""Default model presets for easy loading.

This module defines default model configurations and pre-trained models
that users can easily load by name without needing to configure everything manually.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from uuid import UUID, uuid4

from detectors.models.zoo.configs import HuggingFaceConfig, ModelConfig, PerplexityConfig, GhostbusterConfig


@dataclass
class PretrainedModelInfo:
    """Information about a pre-trained model."""
    name: str
    description: str
    config: ModelConfig
    run_id: Optional[str] = None
    repo_id: Optional[str] = None
    model_class: str = "HuggingFaceDetector"
    tags: List[str] = None
    performance: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Default HuggingFace model configurations
DEFAULT_HUGGINGFACE_CONFIGS = {
    "bert-base-uncased": HuggingFaceConfig(
        model_name="bert-base-uncased",
        model_type="huggingface",
        num_labels=2,
        max_length=512,
        tokenizer_name="bert-base-uncased"
    ),
    "distilbert-base-uncased": HuggingFaceConfig(
        model_name="distilbert-base-uncased",
        model_type="huggingface", 
        num_labels=2,
        max_length=512,
        tokenizer_name="distilbert-base-uncased"
    ),
    "roberta-base": HuggingFaceConfig(
        model_name="roberta-base",
        model_type="huggingface",
        num_labels=2,
        max_length=512,
        tokenizer_name="roberta-base"
    ),
    "deberta-v3-base": HuggingFaceConfig(
        model_name="microsoft/deberta-v3-base",
        model_type="huggingface",
        num_labels=2,
        max_length=512,
        tokenizer_name="microsoft/deberta-v3-base"
    ),
    "electra-base": HuggingFaceConfig(
        model_name="google/electra-base-discriminator",
        model_type="huggingface",
        num_labels=2,
        max_length=512,
        tokenizer_name="google/electra-base-discriminator"
    )
}

# Default Perplexity model configurations
DEFAULT_PERPLEXITY_CONFIGS = {
    "gpt2-perplexity": PerplexityConfig(
        model_name="gpt2",
        model_type="perplexity",
        perplexity_threshold=50.0,
        scaling_factor=0.1
    ),
    "gpt2-medium-perplexity": PerplexityConfig(
        model_name="gpt2-medium",
        model_type="perplexity",
        perplexity_threshold=45.0,
        scaling_factor=0.12
    ),
    "openai-perplexity": PerplexityConfig(
        model_name="gpt-3.5-turbo-instruct",
        model_type="perplexity",
        use_openai=True,
        perplexity_threshold=30.0,
        scaling_factor=0.15
    )
}

# Default Ghostbuster model configurations
DEFAULT_GHOSTBUSTER_CONFIGS = {
    "ghostbuster-gpt2": GhostbusterConfig(
        model_name="ghostbuster-gpt2",
        model_type="ghostbuster",
        estimator_models=["gpt2", "gpt2-medium"],
        feature_extraction_params={"window_size": 20}
    ),
    "ghostbuster-advanced": GhostbusterConfig(
        model_name="ghostbuster-advanced", 
        model_type="ghostbuster",
        estimator_models=["gpt2", "gpt2-medium", "gpt2-large"],
        feature_extraction_params={"window_size": 50, "stride": 10}
    )
}

# Pre-trained model definitions
PRETRAINED_MODELS = {
    # HuggingFace transformer models
    "mango-bert-base": PretrainedModelInfo(
        name="mango-bert-base",
        description="BERT-base model fine-tuned for plagiarism detection on academic texts",
        config=DEFAULT_HUGGINGFACE_CONFIGS["bert-base-uncased"],
        repo_id="MangoTruth/plagiarism-detector-bert-base-academic",
        model_class="HuggingFaceDetector",
        tags=["bert", "academic", "general"],
        performance={
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.91,
            "f1": 0.89
        }
    ),
    
    "mango-distilbert-fast": PretrainedModelInfo(
        name="mango-distilbert-fast",
        description="Fast DistilBERT model optimized for real-time plagiarism detection",
        config=DEFAULT_HUGGINGFACE_CONFIGS["distilbert-base-uncased"],
        repo_id="MangoTruth/plagiarism-detector-distilbert-fast",
        model_class="HuggingFaceDetector",
        tags=["distilbert", "fast", "real-time"],
        performance={
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.87,
            "f1": 0.85
        }
    ),
    
    "mango-roberta-accurate": PretrainedModelInfo(
        name="mango-roberta-accurate",
        description="High-accuracy RoBERTa model for critical plagiarism detection scenarios",
        config=DEFAULT_HUGGINGFACE_CONFIGS["roberta-base"],
        repo_id="MangoTruth/plagiarism-detector-roberta-accurate",
        model_class="HuggingFaceDetector",
        tags=["roberta", "accurate", "critical"],
        performance={
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.94,
            "f1": 0.92
        }
    ),
    
    "mango-deberta-multilingual": PretrainedModelInfo(
        name="mango-deberta-multilingual",
        description="DeBERTa model trained for multilingual plagiarism detection",
        config=DEFAULT_HUGGINGFACE_CONFIGS["deberta-v3-base"],
        repo_id="MangoTruth/plagiarism-detector-deberta-multilingual",
        model_class="HuggingFaceDetector",
        tags=["deberta", "multilingual", "international"],
        performance={
            "accuracy": 0.88,
            "precision": 0.86,
            "recall": 0.90,
            "f1": 0.88
        }
    ),
    
    "mango-electra-efficient": PretrainedModelInfo(
        name="mango-electra-efficient",
        description="ELECTRA model balancing accuracy and efficiency",
        config=DEFAULT_HUGGINGFACE_CONFIGS["electra-base"],
        repo_id="MangoTruth/plagiarism-detector-electra-efficient",
        model_class="HuggingFaceDetector",
        tags=["electra", "efficient", "balanced"],
        performance={
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1": 0.87
        }
    ),
    
    # Perplexity-based models
    "mango-perplexity-gpt2": PretrainedModelInfo(
        name="mango-perplexity-gpt2",
        description="GPT-2 based perplexity model for lightweight AI detection",
        config=DEFAULT_PERPLEXITY_CONFIGS["gpt2-perplexity"],
        repo_id="MangoTruth/plagiarism-detector-perplexity-gpt2",
        model_class="PerplexityModel",
        tags=["perplexity", "gpt2", "lightweight"],
        performance={
            "accuracy": 0.78,
            "precision": 0.75,
            "recall": 0.82,
            "f1": 0.78
        }
    ),
    
    "mango-perplexity-advanced": PretrainedModelInfo(
        name="mango-perplexity-advanced",
        description="Advanced perplexity model using GPT-2 Medium for better accuracy",
        config=DEFAULT_PERPLEXITY_CONFIGS["gpt2-medium-perplexity"],
        repo_id="MangoTruth/plagiarism-detector-perplexity-advanced",
        model_class="PerplexityModel",
        tags=["perplexity", "gpt2-medium", "advanced"],
        performance={
            "accuracy": 0.82,
            "precision": 0.79,
            "recall": 0.85,
            "f1": 0.82
        }
    ),
    
    # Ghostbuster models
    "mango-ghostbuster-basic": PretrainedModelInfo(
        name="mango-ghostbuster-basic",
        description="Ghostbuster model using ensemble of GPT-2 variants for feature extraction",
        config=DEFAULT_GHOSTBUSTER_CONFIGS["ghostbuster-gpt2"],
        repo_id="MangoTruth/plagiarism-detector-ghostbuster-basic",
        model_class="GhostbusterDetector",
        tags=["ghostbuster", "ensemble", "statistical"],
        performance={
            "accuracy": 0.84,
            "precision": 0.82,
            "recall": 0.87,
            "f1": 0.84
        }
    ),
    
    "mango-ghostbuster-advanced": PretrainedModelInfo(
        name="mango-ghostbuster-advanced",
        description="Advanced Ghostbuster model with comprehensive feature extraction",
        config=DEFAULT_GHOSTBUSTER_CONFIGS["ghostbuster-advanced"],
        repo_id="MangoTruth/plagiarism-detector-ghostbuster-advanced",
        model_class="GhostbusterDetector",
        tags=["ghostbuster", "advanced", "comprehensive"],
        performance={
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1": 0.87
        }
    )
}

# Model categories for easy browsing
MODEL_CATEGORIES = {
    "fast": {
        "description": "Models optimized for speed and real-time detection",
        "models": ["mango-distilbert-fast", "mango-electra-efficient", "mango-perplexity-gpt2"]
    },
    "accurate": {
        "description": "Models optimized for highest accuracy",
        "models": ["mango-roberta-accurate", "mango-bert-base", "mango-ghostbuster-advanced"]
    },
    "multilingual": {
        "description": "Models supporting multiple languages",
        "models": ["mango-deberta-multilingual"]
    },
    "general": {
        "description": "General-purpose models for common use cases",
        "models": ["mango-bert-base", "mango-distilbert-fast", "mango-ghostbuster-basic"]
    },
    "academic": {
        "description": "Models specialized for academic text detection",
        "models": ["mango-bert-base", "mango-roberta-accurate"]
    },
    "lightweight": {
        "description": "Lightweight models with minimal resource requirements",
        "models": ["mango-perplexity-gpt2", "mango-distilbert-fast"]
    },
    "transformer": {
        "description": "Models based on transformer architectures",
        "models": ["mango-bert-base", "mango-distilbert-fast", "mango-roberta-accurate", 
                   "mango-deberta-multilingual", "mango-electra-efficient"]
    },
    "statistical": {
        "description": "Models using statistical and ensemble approaches",
        "models": ["mango-perplexity-gpt2", "mango-perplexity-advanced", 
                   "mango-ghostbuster-basic", "mango-ghostbuster-advanced"]
    }
}

# Usage examples and recommendations
USAGE_RECOMMENDATIONS = {
    "research": {
        "description": "For academic research and high-accuracy requirements",
        "recommended_models": ["mango-roberta-accurate", "mango-bert-base", "mango-ghostbuster-advanced"],
        "considerations": "These models provide the highest accuracy but may be slower"
    },
    "production": {
        "description": "For production systems requiring balanced performance",
        "recommended_models": ["mango-electra-efficient", "mango-distilbert-fast", "mango-ghostbuster-basic"],
        "considerations": "Good balance of speed and accuracy for real-world applications"
    },
    "real-time": {
        "description": "For real-time applications where speed is critical",
        "recommended_models": ["mango-distilbert-fast", "mango-perplexity-gpt2"],
        "considerations": "Optimized for low latency, slight trade-off in accuracy"
    },
    "international": {
        "description": "For applications requiring multi-language support",
        "recommended_models": ["mango-deberta-multilingual"],
        "considerations": "Supports multiple languages but may require more resources"
    },
    "low-resource": {
        "description": "For environments with limited computational resources",
        "recommended_models": ["mango-perplexity-gpt2", "mango-distilbert-fast"],
        "considerations": "Minimal GPU/CPU requirements, suitable for edge deployment"
    },
    "high-throughput": {
        "description": "For processing large volumes of text efficiently",
        "recommended_models": ["mango-distilbert-fast", "mango-electra-efficient"],
        "considerations": "Optimized for batch processing and high throughput scenarios"
    }
}


def get_model_info(model_name: str) -> Optional[PretrainedModelInfo]:
    """Get information about a pre-trained model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information or None if not found
    """
    return PRETRAINED_MODELS.get(model_name)


def list_available_models() -> Dict[str, PretrainedModelInfo]:
    """List all available pre-trained models.
    
    Returns:
        Dictionary mapping model names to model information
    """
    return PRETRAINED_MODELS.copy()


def get_models_by_category(category: str) -> List[PretrainedModelInfo]:
    """Get models by category.
    
    Args:
        category: Category name (e.g., 'fast', 'accurate', 'multilingual')
        
    Returns:
        List of model information for the category
    """
    if category not in MODEL_CATEGORIES:
        return []
    
    model_names = MODEL_CATEGORIES[category]["models"]
    return [PRETRAINED_MODELS[name] for name in model_names if name in PRETRAINED_MODELS]


def get_models_by_tag(tag: str) -> List[PretrainedModelInfo]:
    """Get models by tag.
    
    Args:
        tag: Tag to filter by
        
    Returns:
        List of model information with the specified tag
    """
    return [model for model in PRETRAINED_MODELS.values() if tag in model.tags]


def get_recommended_models(use_case: str) -> List[PretrainedModelInfo]:
    """Get recommended models for a specific use case.
    
    Args:
        use_case: Use case name (e.g., 'research', 'production', 'real-time')
        
    Returns:
        List of recommended model information
    """
    if use_case not in USAGE_RECOMMENDATIONS:
        return []
    
    model_names = USAGE_RECOMMENDATIONS[use_case]["recommended_models"]
    return [PRETRAINED_MODELS[name] for name in model_names if name in PRETRAINED_MODELS]


def get_default_config(model_name: str) -> Optional[ModelConfig]:
    """Get default configuration for a base model.
    
    Args:
        model_name: Base model name (e.g., 'bert-base-uncased')
        
    Returns:
        Model configuration or None if not found
    """
    # Check HuggingFace configs first
    if model_name in DEFAULT_HUGGINGFACE_CONFIGS:
        return DEFAULT_HUGGINGFACE_CONFIGS[model_name]
    
    # Check Perplexity configs
    if model_name in DEFAULT_PERPLEXITY_CONFIGS:
        return DEFAULT_PERPLEXITY_CONFIGS[model_name]
    
    # Check Ghostbuster configs  
    if model_name in DEFAULT_GHOSTBUSTER_CONFIGS:
        return DEFAULT_GHOSTBUSTER_CONFIGS[model_name]
    
    return None


def get_models_by_type(model_type: str) -> List[PretrainedModelInfo]:
    """Get models by model type.
    
    Args:
        model_type: Model type (e.g., 'huggingface', 'perplexity', 'ghostbuster')
        
    Returns:
        List of model information for the specified type
    """
    return [model for model in PRETRAINED_MODELS.values() 
            if model.config.model_type == model_type] 