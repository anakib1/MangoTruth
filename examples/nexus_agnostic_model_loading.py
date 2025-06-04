"""Example demonstrating ModelRegistry working with different Nexus implementations.

This script shows how the ModelRegistry is now agnostic to the specific Nexus
implementation and works with any Nexus (Neptune, HuggingFace, etc.).
"""

import logging
from uuid import uuid4, UUID
from pathlib import Path

from detectors.models.zoo.registry import ModelRegistry
from detectors.neptune.nexus import NeptuneNexus
from detectors.models.zoo.hub_nexus import HuggingFaceNexus
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.implementations import HuggingFaceDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_with_neptune_nexus():
    """Demonstrate ModelRegistry with NeptuneNexus."""
    logger.info("=== ModelRegistry with NeptuneNexus ===")
    
    try:
        # Create Neptune nexus (requires NEPTUNE_API_KEY environment variable)
        neptune_nexus = NeptuneNexus()
        
        # Create registry with Neptune nexus
        registry = ModelRegistry(nexus=neptune_nexus)
        
        # Show nexus capabilities
        capabilities = registry._get_nexus_capabilities()
        logger.info(f"Neptune Nexus capabilities: {capabilities}")
        
        # List available models
        models = registry.list_models()
        logger.info(f"Available model types: {list(models.keys())}")
        
        # Load a pretrained model (works without HuggingFace-specific features)
        try:
            model = registry.load_pretrained_model("mango-bert-fast")
            logger.info(f"Successfully loaded model: {type(model).__name__}")
            
            # Test prediction
            text = "This is a test sentence for plagiarism detection."
            probs = model.predict_proba(text)
            labels = model.get_labels()
            logger.info(f"Prediction: {dict(zip(labels, probs))}")
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
        
        # Register and use a custom model
        custom_config = HuggingFaceConfig(
            model_name="bert-base-uncased",
            num_labels=2,
            max_length=256
        )
        
        registry.register_model("custom-bert", HuggingFaceDetector, custom_config)
        custom_model = registry.get_model("custom-bert")
        logger.info(f"Created custom model: {type(custom_model).__name__}")
        
    except Exception as e:
        logger.error(f"Neptune nexus demo failed: {e}")


def demo_with_huggingface_nexus():
    """Demonstrate ModelRegistry with HuggingFaceNexus."""
    logger.info("\n=== ModelRegistry with HuggingFaceNexus ===")
    
    try:
        # Create HuggingFace nexus
        hf_nexus = HuggingFaceNexus()
        
        # Create registry with HuggingFace nexus
        registry = ModelRegistry(nexus=hf_nexus)
        
        # Show nexus capabilities
        capabilities = registry._get_nexus_capabilities()
        logger.info(f"HuggingFace Nexus capabilities: {capabilities}")
        
        # List available models (includes HuggingFace Hub models)
        models = registry.list_models()
        logger.info(f"Available model types: {list(models.keys())}")
        
        # Show HuggingFace-specific features
        if registry._is_huggingface_nexus():
            logger.info("HuggingFace-specific features available!")
            
            # Try to list models from hub
            try:
                hub_models = [k for k in models.keys() if k.startswith('hub_')]
                if hub_models:
                    logger.info(f"Found {len(hub_models)} models from HuggingFace Hub")
                else:
                    logger.info("No HuggingFace Hub models found")
            except Exception as e:
                logger.warning(f"Could not list hub models: {e}")
        
        # Load a pretrained model
        try:
            model = registry.load_pretrained_model("mango-distilbert-fast")
            logger.info(f"Successfully loaded model: {type(model).__name__}")
            
            # Test prediction
            text = "This is a test sentence for plagiarism detection."
            probs = model.predict_proba(text)
            labels = model.get_labels()
            logger.info(f"Prediction: {dict(zip(labels, probs))}")
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            
    except Exception as e:
        logger.error(f"HuggingFace nexus demo failed: {e}")


def demo_without_nexus():
    """Demonstrate ModelRegistry without any Nexus."""
    logger.info("\n=== ModelRegistry without Nexus ===")
    
    # Create registry without nexus
    registry = ModelRegistry(nexus=None)
    
    # Show capabilities
    capabilities = registry._get_nexus_capabilities()
    logger.info(f"No Nexus capabilities: {capabilities}")
    
    # List available models (local only)
    models = registry.list_models()
    logger.info(f"Available model types: {list(models.keys())}")
    
    # Register and use models locally
    custom_config = HuggingFaceConfig(
        model_name="distilbert-base-uncased",
        num_labels=2,
        max_length=256
    )
    
    registry.register_model("local-distilbert", HuggingFaceDetector, custom_config)
    
    # Load a registered model
    model = registry.get_model("local-distilbert")
    logger.info(f"Created local model: {type(model).__name__}")
    
    # Test prediction
    text = "This is a test sentence for plagiarism detection."
    probs = model.predict_proba(text)
    labels = model.get_labels()
    logger.info(f"Prediction: {dict(zip(labels, probs))}")
    
    # Try to load pretrained model (should work for models without run_ids)
    try:
        # This should work because it doesn't require loading weights
        pretrained_model = registry.load_pretrained_model("mango-bert-fast")
        logger.info(f"Loaded pretrained model (no weights): {type(pretrained_model).__name__}")
    except Exception as e:
        logger.info(f"Expected: Cannot load weights without nexus: {e}")


def demo_run_id_loading():
    """Demonstrate loading models by run ID."""
    logger.info("\n=== Loading Models by Run ID ===")
    
    # Example run IDs (these would come from actual training runs)
    example_run_ids = [
        "12345678-1234-5678-9abc-123456789abc",
        "87654321-4321-8765-cba9-987654321cba"
    ]
    
    for nexus_class, nexus_name in [(NeptuneNexus, "Neptune"), (HuggingFaceNexus, "HuggingFace")]:
        try:
            logger.info(f"--- Testing with {nexus_name} ---")
            
            # Create nexus and registry
            if nexus_name == "Neptune":
                nexus = NeptuneNexus()
            else:
                nexus = HuggingFaceNexus()
                
            registry = ModelRegistry(nexus=nexus)
            
            # Try to load by run ID
            for run_id_str in example_run_ids:
                try:
                    model = registry.load_pretrained_model(run_id_str)
                    logger.info(f"Successfully loaded model from run {run_id_str}")
                    break
                except Exception as e:
                    logger.debug(f"Could not load run {run_id_str}: {e}")
            else:
                logger.info(f"No valid run IDs found for {nexus_name}")
                
        except Exception as e:
            logger.warning(f"Could not test {nexus_name}: {e}")


def demo_discovery():
    """Demonstrate model discovery capabilities."""
    logger.info("\n=== Model Discovery ===")
    
    # Test with different nexus types
    for nexus_class, nexus_name in [(None, "No Nexus"), (NeptuneNexus, "Neptune"), (HuggingFaceNexus, "HuggingFace")]:
        try:
            logger.info(f"--- Discovery with {nexus_name} ---")
            
            # Create registry
            if nexus_class is None:
                registry = ModelRegistry(nexus=None)
            else:
                nexus = nexus_class()
                registry = ModelRegistry(nexus=nexus)
            
            # Discover models
            discovery = registry.discover_models()
            
            logger.info(f"Nexus type: {discovery['nexus_type']}")
            logger.info(f"Registered models: {len(discovery['registered'])}")
            logger.info(f"Pretrained models: {len(discovery['pretrained'])}")
            logger.info(f"Categories: {len(discovery['categories'])}")
            logger.info(f"Use cases: {len(discovery['use_cases'])}")
            
            # Show capabilities
            capabilities = discovery['nexus_capabilities']
            logger.info(f"Capabilities: {capabilities}")
            
        except Exception as e:
            logger.warning(f"Discovery with {nexus_name} failed: {e}")


def main():
    """Run all demonstrations."""
    logger.info("Demonstrating Nexus-agnostic ModelRegistry")
    logger.info("=" * 60)
    
    # Demo without nexus (always works)
    demo_without_nexus()
    
    # Demo with Neptune (requires API key)
    demo_with_neptune_nexus()
    
    # Demo with HuggingFace (requires token)
    demo_with_huggingface_nexus()
    
    # Demo run ID loading
    demo_run_id_loading()
    
    # Demo discovery
    demo_discovery()
    
    logger.info("\n" + "=" * 60)
    logger.info("All demonstrations completed!")


if __name__ == "__main__":
    main() 