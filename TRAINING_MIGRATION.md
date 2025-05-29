# Training Migration Guide

This document outlines the migration from the old training scripts to the new unified training architecture.

## Architecture Changes

### Old Approach (Scripts Folder)
- `train_roberta.py`: Direct HuggingFace training with custom loops
- `train_perplexity.py`: Manual threshold optimization and model training  
- `train_ghostbuster.py`: Direct sklearn training with feature extraction

### New Approach (Training Module)
- **Unified Interface**: All models implement `ITrainable` interface
- **Pluggable Trainers**: Dedicated trainer classes for each model type
- **Configuration Management**: Structured config classes for reproducibility
- **Consistent Workflow**: Same API across all model types

## Migration Mapping

### RoBERTa Training

**OLD**: `detectors/scripts/train_roberta.py`
```python
# Direct transformers usage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Manual setup and training loop
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

**NEW**: `detectors/scripts/train_roberta_new.py`
```python
# Using new architecture
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.training import HuggingFaceTrainer, HuggingFaceTrainingConfig

# Structured approach
model = TrainableHuggingFaceDetector(config)
trainer = HuggingFaceTrainer(model, training_config)
result = trainer.train(train_dataset, validation_dataset)
```

### Perplexity Training

**OLD**: `detectors/scripts/train_perplexity.py`
```python
# Manual threshold optimization
def train_threshold(X, y):
    thresholds = np.sort(X)
    max_f1 = 0
    for C in thresholds:
        predictions = (X > C).astype(int)
        f1 = f1_score(y, predictions)
        # ... manual optimization

# Direct model instantiation
model = PerplexityModel(model_handle=args.model_handle)
```

**NEW**: `detectors/scripts/train_perplexity_new.py`
```python
# Automated training pipeline
from detectors.models.zoo.trainable_implementations import TrainablePerplexityModel
from detectors.training import PerplexityTrainer, PerplexityTrainingConfig

model = TrainablePerplexityModel(model_handle=args.perplexity_model)
trainer = PerplexityTrainer(model, training_config)
result = trainer.train(train_dataset, validation_dataset)
```

### Ghostbuster Training

**OLD**: `detectors/scripts/train_ghostbuster.py`
```python
# Manual feature extraction and model training
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Feature extraction loop
feats = []
for i in tqdm(range(len(X))):
    feats.append(extract_features([model.get_text_log_proba(X[i])[1] for model in models]))

# Manual sklearn training
model = make_pipeline(StandardScaler(), CalibratedClassifierCV(...))
model.fit(X_train, y_train)
```

**NEW**: `detectors/scripts/train_ghostbuster_new.py`
```python
# Automated pipeline
from detectors.models.zoo.trainable_implementations import TrainableGhostbusterDetector
from detectors.training import GhostbusterTrainer, GhostbusterTrainingConfig

model = TrainableGhostbusterDetector()
trainer = GhostbusterTrainer(model, training_config)
result = trainer.train(train_dataset, validation_dataset)
```

## Key Benefits

### 1. Unified Interface
- All models implement the same `ITrainable` interface
- Consistent methods: `train_mode()`, `compute_loss()`, `save_checkpoint()`, etc.
- Easy to swap between different model types

### 2. Configuration Management
- Structured configuration classes
- Serializable and reproducible settings
- Type hints and validation

### 3. Standardized Training Results
- `TrainingResult` class with consistent metrics
- Training/validation/test conclusions
- Metadata and duration tracking

### 4. Better Error Handling
- Proper exception handling in training loops
- Graceful fallbacks for missing components
- Comprehensive logging

### 5. Dataset Integration
- Uses unified `Dataset` and `TextSample` classes
- Consistent data processing across all models
- Better support for different data sources

## Configuration Examples

### HuggingFace Training Config
```python
config = HuggingFaceTrainingConfig(
    model_name='roberta-base',
    max_length=512,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    warmup_steps=100,
    weight_decay=0.01,
    output_dir='./training_output',
    save_strategy='epoch'
)
```

### Perplexity Training Config
```python
config = PerplexityTrainingConfig(
    model_handle='babbage-002',
    validation_split=0.3,
    batch_size=4,
    output_dir='./training_output'
)
```

### Ghostbuster Training Config
```python
config = GhostbusterTrainingConfig(
    tokenizer_handle='gugarosa/cl100k_base',
    llm_handles=['babbage-002'],
    max_length=15000,
    validation_split=0.3,
    classifier_config={'C': 1, 'max_iter': 1000}
)
```

## Migration Steps

1. **Update Imports**: Replace direct model imports with trainable implementations
2. **Create Config**: Use structured configuration classes instead of argument parsing
3. **Use Trainer**: Replace manual training loops with trainer classes
4. **Handle Results**: Use `TrainingResult` for consistent output handling
5. **Save Models**: Use trainer's save/load methods for persistence

## Backward Compatibility

- Old scripts are preserved for reference
- New scripts follow naming convention: `*_new.py`
- Both approaches can coexist during transition period
- Trained models are compatible between old and new approaches

## Future Extensions

The new architecture makes it easy to:
- Add new model types by implementing `ITrainable`
- Create custom trainers for specialized training procedures
- Add new configuration options without breaking existing code
- Implement advanced training features (early stopping, learning rate scheduling, etc.)

## Example Usage

```python
# Load and prepare data
dataset = load_dataset('anakib1/mango-truth', 'xlsum')
train_dataset = process_huggingface_dataset(dataset['train'], 1000)

# Create model and trainer
model = TrainableHuggingFaceDetector(model_config)
trainer = HuggingFaceTrainer(model, training_config)

# Train
result = trainer.train(train_dataset)

# Access results
print(f"Training time: {result.training_duration_seconds:.2f}s")
print(f"Validation F1: {result.validation_conclusion.metrics.f1:.4f}")

# Save model
trainer.save_model('./my_model')
```

This new architecture provides a more maintainable, extensible, and user-friendly approach to training detection models while preserving all the functionality of the original scripts. 