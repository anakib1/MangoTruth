# Model Registry Usage Guide

The ModelRegistry provides a unified interface for managing and loading plagiarism detection models. It works with any Nexus implementation (Neptune, HuggingFace Hub, etc.) and provides fallback mechanisms when specific features aren't available.

## Key Features

- **Nexus-Agnostic**: Works with any Nexus implementation (NeptuneNexus, HuggingFaceNexus, etc.)
- **Easy Model Loading**: Load models by name, run ID, or repository ID  
- **Default Presets**: Pre-configured models for common use cases
- **Smart Fallbacks**: Graceful degradation when features aren't available
- **Discovery**: Browse and explore available models
- **Runtime Configuration**: Override model settings at load time

## Basic Usage

### Without Nexus (Local Only)

```python
from detectors.models.zoo.registry import ModelRegistry
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.implementations import HuggingFaceDetector

# Create registry without nexus
registry = ModelRegistry(nexus=None)

# Register a custom model
config = HuggingFaceConfig(
    model_name="bert-base-uncased",
    num_labels=2,
    max_length=512
)

registry.register_model("my-bert", HuggingFaceDetector, config)

# Load and use the model
model = registry.get_model("my-bert")
probs = model.predict_proba("This is a test sentence.")
print(f"Predictions: {dict(zip(model.get_labels(), probs))}")
```

### With Neptune Nexus

```python
from detectors.models.zoo.registry import ModelRegistry
from detectors.neptune.nexus import NeptuneNexus

# Create Neptune nexus (requires NEPTUNE_API_KEY)
nexus = NeptuneNexus()
registry = ModelRegistry(nexus=nexus)

# Load a pretrained model by name
model = registry.load_pretrained_model("mango-bert-fast")

# Or load by run ID
model = registry.load_pretrained_model("12345678-1234-5678-9abc-123456789abc")

# Check nexus capabilities
capabilities = registry._get_nexus_capabilities()
print(f"Neptune capabilities: {capabilities}")
```

### With HuggingFace Hub Nexus

```python
from detectors.models.zoo.registry import ModelRegistry
from detectors.models.zoo.hub_nexus import HuggingFaceNexus

# Create HuggingFace nexus (requires HF_TOKEN)
nexus = HuggingFaceNexus()
registry = ModelRegistry(nexus=nexus)

# Load from HuggingFace Hub repository
model = registry.load_pretrained_model("anakib1/plagiarism-detector-12345")

# Access HuggingFace-specific features
if registry._is_huggingface_nexus():
    models = registry.list_models()
    hub_models = [k for k in models.keys() if k.startswith('hub_')]
    print(f"Found {len(hub_models)} models from HuggingFace Hub")
```

## Model Loading Options

### By Predefined Name

```python
# Load fast models
fast_model = registry.load_pretrained_model("mango-distilbert-fast")

# Load accurate models  
accurate_model = registry.load_pretrained_model("mango-roberta-accurate")

# Load multilingual models
multilingual_model = registry.load_pretrained_model("mango-bert-multilingual")
```

### By Run ID (UUID)

```python
# Direct run ID loading (works with any Nexus)
run_id = "12345678-1234-5678-9abc-123456789abc"
model = registry.load_pretrained_model(run_id)
```

### By Repository ID (HuggingFace only)

```python
# HuggingFace Hub repository (requires HuggingFaceNexus)
repo_id = "anakib1/plagiarism-detector-12345678"
model = registry.load_pretrained_model(repo_id)
```

### With Runtime Configuration

```python
# Override configuration at load time
model = registry.load_pretrained_model(
    "mango-bert-fast",
    max_length=256,  # Override default 512
    batch_size=32,   # Add new parameter
    device="cuda"    # Override device
)
```

## Nexus Compatibility

The ModelRegistry automatically adapts to the capabilities of the provided Nexus:

### Core Nexus Interface (All Implementations)

- `load_run_weights(run_id)` - Load model weights
- `store_run_weights(run_id, content)` - Store model weights

### Extended Features (Implementation-Specific)

**NeptuneNexus:**
- ✅ Load/store weights
- ✅ Training conclusions
- ❌ Model metadata
- ❌ Hub listing

**HuggingFaceNexus:**
- ✅ Load/store weights  
- ✅ Model metadata
- ✅ Hub listing
- ✅ Repository management

## Discovery and Browsing

### Discover Available Models

```python
discovery = registry.discover_models()

print(f"Nexus type: {discovery['nexus_type']}")
print(f"Registered models: {discovery['registered']}")
print(f"Pretrained models: {discovery['pretrained']}")
print(f"Capabilities: {discovery['nexus_capabilities']}")
```

### Browse by Category

```python
# Get fast models
fast_models = registry.get_models_by_category("fast")
print(f"Fast models: {[m.name for m in fast_models]}")

# Get accurate models
accurate_models = registry.get_models_by_category("accurate") 
print(f"Accurate models: {[m.name for m in accurate_models]}")
```

### Get Recommendations

```python
# Get models for production use
production_models = registry.get_recommended_models("production")
print(f"Production models: {[m.name for m in production_models]}")

# Get models for research
research_models = registry.get_recommended_models("research")
print(f"Research models: {[m.name for m in research_models]}")
```

## Error Handling and Fallbacks

The registry provides graceful fallbacks when features aren't available:

```python
try:
    # Try to load with weights
    model = registry.load_pretrained_model("mango-bert-fast")
except ValueError as e:
    if "nexus" in str(e).lower():
        # Fallback: load without weights
        registry_no_nexus = ModelRegistry(nexus=None)
        model = registry_no_nexus.load_pretrained_model("mango-bert-fast")
        print("Loaded model without pretrained weights")
```

## Switching Between Nexus Implementations

You can change the Nexus at runtime:

```python
# Start with Neptune
registry = ModelRegistry(nexus=NeptuneNexus())

# Switch to HuggingFace
registry.set_nexus(HuggingFaceNexus())

# Remove nexus for local-only operation
registry.set_nexus(None)
```

## Best Practices

### 1. Check Capabilities Before Use

```python
capabilities = registry._get_nexus_capabilities()

if capabilities["metadata"]:
    # Use metadata-dependent features
    pass

if capabilities["huggingface_features"]:
    # Use HuggingFace-specific features
    pass
```

### 2. Handle Missing Models Gracefully

```python
try:
    model = registry.load_pretrained_model("specific-model")
except ValueError:
    # Fallback to a known working model
    model = registry.load_pretrained_model("mango-bert-fast")
```

### 3. Use Appropriate Models for Use Cases

```python
# For production (fast inference)
model = registry.load_pretrained_model("mango-distilbert-fast")

# For research (high accuracy)  
model = registry.load_pretrained_model("mango-roberta-accurate")

# For multilingual content
model = registry.load_pretrained_model("mango-bert-multilingual")
```

## Custom Model Registration

```python
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.implementations import HuggingFaceDetector

# Define custom configuration
config = HuggingFaceConfig(
    model_name="distilbert-base-uncased",
    num_labels=2,
    max_length=256,
    dropout_rate=0.1
)

# Register with metadata
registry.register_custom_model(
    name="custom-distilbert",
    model_class=HuggingFaceDetector,
    config=config,
    description="Custom DistilBERT for fast inference",
    tags=["fast", "lightweight", "custom"]
)

# Use the registered model
model = registry.get_model("custom-distilbert")
```

This design ensures the ModelRegistry works seamlessly with any Nexus implementation while providing enhanced features when they're available. 