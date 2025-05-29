# MangoTruth Model Zoo

The MangoTruth Model Zoo provides an integrated system for managing, discovering, and using plagiarism detection models. It combines a powerful `ModelRegistry` with `HuggingFaceNexus` for seamless model storage and retrieval from HuggingFace Hub.

## Features

- 🚀 **Easy Model Loading**: Load pre-trained models with simple names
- 🔍 **Model Discovery**: Browse models by category, use case, and tags
- 💾 **HuggingFace Integration**: Store and share models on HuggingFace Hub
- 🎯 **Smart Recommendations**: Get model suggestions based on your use case
- ⚙️ **Flexible Configuration**: Override model settings at runtime
- 🔄 **Multiple Model Types**: Support for HuggingFace, Perplexity, and Ghostbuster models

## Quick Start

### Basic Setup

```python
from detectors.models.zoo import ModelRegistry, HuggingFaceNexus

# Set up the integrated system
nexus = HuggingFaceNexus(namespace="MangoTruth")
registry = ModelRegistry(nexus=nexus)
```

### Load a Pre-trained Model

```python
# Load by name - automatically downloads and configures
model = registry.load_pretrained_model("mango-distilbert-fast")

# Make predictions
text = "This is a sample text for plagiarism detection."
prediction = model.predict_proba(text)
labels = model.get_labels()

for label, prob in zip(labels, prediction):
    print(f"{label}: {prob:.3f}")
```

## Available Models

### By Category

#### Fast Models
- `mango-distilbert-fast`: DistilBERT optimized for real-time detection
- `mango-perplexity-gpt2`: Lightweight GPT-2 based perplexity model
- `mango-electra-efficient`: Efficient ELECTRA model

#### Accurate Models
- `mango-roberta-accurate`: High-accuracy RoBERTa model
- `mango-bert-base`: Fine-tuned BERT for academic texts
- `mango-ghostbuster-advanced`: Advanced ensemble model

#### Specialized Models
- `mango-deberta-multilingual`: Multi-language support
- `mango-perplexity-advanced`: Advanced perplexity detection
- `mango-ghostbuster-basic`: Statistical feature-based detection

### By Use Case

```python
# Get recommendations for your use case
research_models = registry.get_recommended_models("research")
production_models = registry.get_recommended_models("production") 
real_time_models = registry.get_recommended_models("real-time")
```

## Model Discovery

### Browse by Category
```python
# Discover models by category
fast_models = registry.get_models_by_category("fast")
accurate_models = registry.get_models_by_category("accurate")
lightweight_models = registry.get_models_by_category("lightweight")

# Show all available categories
discovery = registry.discover_models()
print("Categories:", list(discovery["categories"].keys()))
```

### Search by Tags
```python
from detectors.models.zoo import get_models_by_tag

# Find models with specific tags
transformer_models = get_models_by_tag("transformer")
statistical_models = get_models_by_tag("statistical")
```

## Advanced Usage

### Custom Configuration
```python
# Override model settings at load time
custom_model = registry.load_pretrained_model(
    "mango-bert-base",
    max_length=256,      # Shorter sequences
    batch_size=32,       # Custom batch size
    device="cpu"         # Force CPU usage
)
```

### Register Custom Models
```python
from detectors.models.zoo import HuggingFaceConfig, HuggingFaceDetector

# Create custom configuration
config = HuggingFaceConfig(
    model_name="your-custom-model",
    num_labels=3,
    max_length=384
)

# Register with metadata
registry.register_custom_model(
    name="my-custom-detector",
    model_class=HuggingFaceDetector,
    config=config,
    description="Custom multi-class detector",
    tags=["custom", "multiclass"]
)
```

### Training Integration
```python
from detectors.models.zoo import TrainableHuggingFaceDetector
from uuid import uuid4

# Create trainable model
config = HuggingFaceConfig(model_name="distilbert-base-uncased")
model = TrainableHuggingFaceDetector(config)

# After training, upload to HuggingFace Hub
run_id = uuid4()
weights = model.store_weights()
nexus.store_run_weights(run_id, weights)

# Later, load the trained model
loaded_weights = nexus.load_run_weights(run_id)
new_model = TrainableHuggingFaceDetector(config)
new_model.load_weights(loaded_weights)
```

## HuggingFace Hub Integration

### Setup
```python
# Initialize with your HuggingFace token
nexus = HuggingFaceNexus(
    namespace="YourOrganization",
    token="your_hf_token",  # or set HF_TOKEN environment variable
    private=False           # Create public repositories
)
```

### Upload Models
```python
from detectors.metrics import Conclusion, SplitConclusion, Metrics

# Create conclusion with metrics
conclusion = Conclusion(
    weights=model.store_weights(),
    detector_handle="my-detector",
    datasets=["academic-corpus"],
    train_conclusion=SplitConclusion(metrics=Metrics(accuracy=0.89, f1=0.87)),
    validation_conclusion=SplitConclusion(metrics=Metrics(accuracy=0.85, f1=0.83))
)

# Upload with metadata
nexus.conclude_run(run_id, conclusion, extra_data={"info": "Custom training"})
```

### Browse Available Models
```python
# List models in your namespace
available_models = nexus.list_available_models()
for model in available_models:
    print(f"Model: {model['model_id']}")
    print(f"Downloads: {model['downloads']}")
```

## Model Types

### HuggingFace Models
```python
from detectors.models.zoo import HuggingFaceConfig

config = HuggingFaceConfig(
    model_name="bert-base-uncased",
    num_labels=2,
    max_length=512,
    tokenizer_name="bert-base-uncased"
)
```

### Perplexity Models
```python
from detectors.models.zoo.configs import PerplexityConfig

config = PerplexityConfig(
    model_name="gpt2",
    perplexity_threshold=50.0,
    scaling_factor=0.1,
    use_openai=False
)
```

### Ghostbuster Models
```python
from detectors.models.zoo.configs import GhostbusterConfig

config = GhostbusterConfig(
    model_name="ghostbuster-ensemble",
    estimator_models=["gpt2", "gpt2-medium"],
    feature_extraction_params={"window_size": 20}
)
```

## Performance Comparison

| Model | Type | Accuracy | F1 Score | Speed | Use Case |
|-------|------|----------|----------|-------|----------|
| mango-roberta-accurate | Transformer | 0.92 | 0.92 | Slow | Research |
| mango-bert-base | Transformer | 0.89 | 0.89 | Medium | General |
| mango-distilbert-fast | Transformer | 0.85 | 0.85 | Fast | Production |
| mango-ghostbuster-advanced | Statistical | 0.87 | 0.87 | Medium | Ensemble |
| mango-perplexity-gpt2 | Statistical | 0.78 | 0.78 | Very Fast | Lightweight |

## Configuration Reference

### HuggingFaceConfig
- `model_name`: HuggingFace model identifier
- `tokenizer_name`: Tokenizer to use (defaults to model_name)
- `num_labels`: Number of classification labels
- `max_length`: Maximum sequence length
- `batch_size`: Batch size for inference
- `device`: Computing device ("cuda", "cpu", "mps")

### PerplexityConfig
- `model_name`: Language model for perplexity calculation
- `perplexity_threshold`: Threshold for classification
- `scaling_factor`: Scaling factor for sigmoid function
- `use_openai`: Whether to use OpenAI API models

### GhostbusterConfig
- `estimator_models`: List of models for feature extraction
- `feature_extraction_params`: Parameters for feature extraction
- `classifier_params`: Parameters for the ML classifier

## Best Practices

### Choosing Models

1. **Research/Academic**: Use `mango-roberta-accurate` or `mango-bert-base`
2. **Production Systems**: Use `mango-electra-efficient` or `mango-distilbert-fast`
3. **Real-time Applications**: Use `mango-distilbert-fast` or `mango-perplexity-gpt2`
4. **Multilingual**: Use `mango-deberta-multilingual`
5. **Low Resources**: Use `mango-perplexity-gpt2`

### Performance Optimization

```python
# For batch processing
model = registry.load_pretrained_model(
    "mango-distilbert-fast",
    batch_size=64,
    max_length=256
)

# For single predictions
model = registry.load_pretrained_model(
    "mango-perplexity-gpt2",
    device="cpu"  # Use CPU for single predictions
)
```

### Model Deployment

```python
# Save configuration for deployment
config = registry.get_config("mango-distilbert-fast")
config.save("model_config.json")

# Load in production
from detectors.models.zoo.configs import HuggingFaceConfig
config = HuggingFaceConfig.load("model_config.json")
model = registry.get_model("huggingface", config=config)
```

## Error Handling

```python
try:
    model = registry.load_pretrained_model("non-existent-model")
except ValueError as e:
    print(f"Model not found: {e}")
    # Fall back to base model
    model = registry.get_model("distilbert-base-uncased")

try:
    # Upload requires HuggingFace token
    nexus.store_run_weights(run_id, weights)
except Exception as e:
    print(f"Upload failed: {e}")
    # Save locally instead
    with open(f"model_{run_id}.pkl", "wb") as f:
        f.write(weights)
```

## Contributing

To add new models to the zoo:

1. Create model configuration in `default_models.py`
2. Add to appropriate categories
3. Include performance metrics
4. Upload trained weights to HuggingFace Hub
5. Update documentation

## License

This project is part of the MangoTruth plagiarism detection framework. See the main repository for license information. 