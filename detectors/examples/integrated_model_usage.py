"""Example usage of the integrated ModelRegistry and Nexus system.

This script demonstrates how to:
1. Set up the integrated system with HuggingFace Hub storage
2. Load pre-trained models using simple names
3. Discover and browse available models
4. Train and upload new models
5. Use different model presets for different use cases
"""

import logging
from uuid import uuid4

from detectors.models.zoo import (
    ModelRegistry, 
    HuggingFaceNexus,
    HuggingFaceConfig,
    HuggingFaceDetector,
    TrainableHuggingFaceDetector,
    get_model_info,
    get_models_by_category,
    get_recommended_models
)
from detectors.data.datasets import Dataset, TextSample

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_integrated_system():
    """Set up the integrated ModelRegistry and HuggingFace Nexus system."""
    logger.info("Setting up integrated system...")
    
    # Initialize HuggingFace Nexus for model storage
    nexus = HuggingFaceNexus(
        namespace="MangoTruth",  # Your HuggingFace organization
        cache_dir="./cache/models"
    )
    
    # Initialize ModelRegistry with Nexus integration
    registry = ModelRegistry(nexus=nexus)
    
    # Register additional model implementations
    registry.register_model("huggingface", HuggingFaceDetector)
    registry.register_model("trainable-huggingface", TrainableHuggingFaceDetector)
    
    return registry, nexus


def demonstrate_model_discovery(registry: ModelRegistry):
    """Demonstrate model discovery and browsing capabilities."""
    logger.info("=== Model Discovery ===")
    
    # List all available models
    all_models = registry.list_models()
    logger.info(f"Total available models: {len(all_models)}")
    
    # Show pretrained models
    pretrained_models = registry.list_pretrained_models()
    logger.info(f"Pre-trained models: {len(pretrained_models)}")
    for name, info in pretrained_models.items():
        logger.info(f"  - {name}: {info.description}")
        if info.performance:
            logger.info(f"    Performance: {info.performance}")
    
    # Browse by category
    logger.info("\n=== Models by Category ===")
    fast_models = registry.get_models_by_category("fast")
    logger.info(f"Fast models: {[m.name for m in fast_models]}")
    
    accurate_models = registry.get_models_by_category("accurate")
    logger.info(f"Accurate models: {[m.name for m in accurate_models]}")
    
    # Get recommendations for use cases
    logger.info("\n=== Recommendations ===")
    research_models = registry.get_recommended_models("research")
    logger.info(f"Research recommended: {[m.name for m in research_models]}")
    
    production_models = registry.get_recommended_models("production")
    logger.info(f"Production recommended: {[m.name for m in production_models]}")


def demonstrate_easy_model_loading(registry: ModelRegistry):
    """Demonstrate easy model loading with default presets."""
    logger.info("=== Easy Model Loading ===")
    
    try:
        # Load a fast model for real-time use
        logger.info("Loading fast model for real-time use...")
        fast_model = registry.load_pretrained_model("mango-distilbert-fast")
        
        # Test prediction
        test_text = "This is a sample text for plagiarism detection."
        prediction = fast_model.predict_proba(test_text)
        labels = fast_model.get_labels()
        
        logger.info(f"Prediction results:")
        for label, prob in zip(labels, prediction):
            logger.info(f"  {label}: {prob:.3f}")
    
    except Exception as e:
        logger.warning(f"Could not load pre-trained model (expected in demo): {e}")
        
        # Fall back to base model
        logger.info("Loading base model instead...")
        base_config = HuggingFaceConfig(
            model_name="distilbert-base-uncased",
            num_labels=2,
            max_length=256
        )
        registry.register_model("demo-distilbert", HuggingFaceDetector, base_config)
        demo_model = registry.get_model("demo-distilbert")
        
        prediction = demo_model.predict_proba(test_text)
        labels = demo_model.get_labels()
        
        logger.info(f"Demo prediction results:")
        for label, prob in zip(labels, prediction):
            logger.info(f"  {label}: {prob:.3f}")


def demonstrate_model_comparison(registry: ModelRegistry):
    """Demonstrate comparing different model presets."""
    logger.info("=== Model Comparison ===")
    
    # Get model information for comparison
    fast_info = get_model_info("mango-distilbert-fast")
    accurate_info = get_model_info("mango-roberta-accurate")
    
    if fast_info and accurate_info:
        logger.info("Comparing Fast vs Accurate models:")
        logger.info(f"Fast Model ({fast_info.name}):")
        logger.info(f"  Description: {fast_info.description}")
        logger.info(f"  Performance: {fast_info.performance}")
        logger.info(f"  Tags: {fast_info.tags}")
        
        logger.info(f"\nAccurate Model ({accurate_info.name}):")
        logger.info(f"  Description: {accurate_info.description}")
        logger.info(f"  Performance: {accurate_info.performance}")
        logger.info(f"  Tags: {accurate_info.tags}")
        
        # Performance comparison
        if fast_info.performance and accurate_info.performance:
            fast_f1 = fast_info.performance.get('f1', 0)
            accurate_f1 = accurate_info.performance.get('f1', 0)
            logger.info(f"\nF1 Score Comparison:")
            logger.info(f"  Fast model: {fast_f1:.3f}")
            logger.info(f"  Accurate model: {accurate_f1:.3f}")
            logger.info(f"  Accuracy gain: {accurate_f1 - fast_f1:.3f}")


def demonstrate_training_integration(registry: ModelRegistry, nexus: HuggingFaceNexus):
    """Demonstrate training a new model and uploading to HuggingFace Hub."""
    logger.info("=== Training Integration Demo ===")
    
    # This is a simplified demo - in practice you'd have real training data
    logger.info("Note: This is a simplified demo of the training integration")
    
    # Create a trainable model
    config = HuggingFaceConfig(
        model_name="distilbert-base-uncased",
        num_labels=2,
        max_length=256
    )
    
    model = TrainableHuggingFaceDetector(config)
    
    # Simulate training completion
    run_id = uuid4()
    logger.info(f"Simulated training run: {run_id}")
    
    # Store model weights (in practice, this would be done by the trainer)
    model_weights = model.store_weights()
    
    try:
        # Upload to HuggingFace Hub
        logger.info("Uploading model to HuggingFace Hub...")
        nexus.store_run_weights(run_id, model_weights)
        logger.info(f"Model uploaded successfully to repository: {nexus._get_repo_id(run_id)}")
        
        # Later, load the model back
        logger.info("Loading model back from HuggingFace Hub...")
        loaded_weights = nexus.load_run_weights(run_id)
        
        # Create new model instance and load weights
        new_model = TrainableHuggingFaceDetector(config)
        new_model.load_weights(loaded_weights)
        logger.info("Model loaded successfully from HuggingFace Hub")
        
    except Exception as e:
        logger.warning(f"Could not upload/download model (expected without HF token): {e}")


def demonstrate_custom_model_registration(registry: ModelRegistry):
    """Demonstrate registering custom models with metadata."""
    logger.info("=== Custom Model Registration ===")
    
    # Create custom configuration
    custom_config = HuggingFaceConfig(
        model_name="bert-base-cased",
        num_labels=3,  # Multi-class detection
        max_length=384,
        tokenizer_name="bert-base-cased"
    )
    
    # Register custom model with metadata
    registry.register_custom_model(
        name="custom-bert-multiclass",
        model_class=HuggingFaceDetector,
        config=custom_config,
        description="Custom BERT model for multi-class plagiarism detection",
        tags=["bert", "multiclass", "custom"]
    )
    
    # Use the custom model
    custom_model = registry.get_model("custom-bert-multiclass")
    logger.info(f"Custom model labels: {custom_model.get_labels()}")
    logger.info(f"Custom model config: {custom_model.config.to_dict()}")


def demonstrate_configuration_override(registry: ModelRegistry):
    """Demonstrate overriding model configurations at runtime."""
    logger.info("=== Configuration Override ===")
    
    # Load model with custom parameters
    try:
        # Override max_length and batch_size
        custom_model = registry.load_pretrained_model(
            "mango-distilbert-fast",
            max_length=128,  # Shorter sequences for faster processing
            batch_size=64    # Larger batch for throughput
        )
        logger.info("Loaded model with custom configuration")
        logger.info(f"Max length: {custom_model.config.max_length}")
        logger.info(f"Batch size: {custom_model.config.batch_size}")
        
    except Exception as e:
        logger.warning(f"Could not load pretrained model: {e}")
        
        # Demonstrate with base model
        base_model = registry.get_model(
            "distilbert-base-uncased",
            max_length=128,
            batch_size=64
        )
        logger.info("Loaded base model with custom configuration")


def main():
    """Main demonstration function."""
    logger.info("Starting MangoTruth Model Registry and Nexus Integration Demo")
    
    # Set up the integrated system
    registry, nexus = setup_integrated_system()
    
    # Demonstrate various capabilities
    demonstrate_model_discovery(registry)
    demonstrate_easy_model_loading(registry)
    demonstrate_model_comparison(registry)
    demonstrate_custom_model_registration(registry)
    demonstrate_configuration_override(registry)
    demonstrate_training_integration(registry, nexus)
    
    # Show final system state
    logger.info("\n=== Final System State ===")
    discovery = registry.discover_models()
    logger.info(f"Registered models: {discovery['registered']}")
    logger.info(f"Pretrained models: {discovery['pretrained']}")
    logger.info(f"Available categories: {list(discovery['categories'].keys())}")
    logger.info(f"Available use cases: {list(discovery['use_cases'].keys())}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() 