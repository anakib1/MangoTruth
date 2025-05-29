# Training Architecture Documentation

This document outlines the improved training architecture for the detectors package, following industry best practices and DRY principles.

## Overview

The training system has been refactored to eliminate code duplication, improve maintainability, and provide a consistent interface across different model types and training approaches.

## Architecture Components

### 1. Core Interfaces (`detectors/training/interfaces.py`)

- **`ITrainable`**: Interface for trainable models
- **`ITrainer`**: Interface for trainer implementations
- **`TrainingConfig`**: Base configuration class
- **`TrainingResult`**: Standardized training result format

### 2. Configuration Management (`detectors/training/configs.py`)

- **`HuggingFaceTrainingConfig`**: Configuration for transformer models
- **`ClassicalMLTrainingConfig`**: Configuration for traditional ML models
- Supports serialization/deserialization to/from JSON/YAML
- Command-line override support

### 3. Trainer Implementations (`detectors/training/trainers/`)

- **`BaseTrainer`**: Common functionality for all trainers
- **`HuggingFaceTrainer`**: Specialized for transformer models
- **`PerplexityTrainer`**: For perplexity-based models
- **`GhostbusterTrainer`**: For Ghostbuster models

### 4. Factory Pattern (`detectors/training/factories/`)

#### TrainerFactory
Centralizes trainer creation logic:
```python
from detectors.training.factories import TrainerFactory, TrainerType

trainer = TrainerFactory.create_trainer(
    TrainerType.HUGGINGFACE, 
    model, 
    config
)
```

#### ModelFactory (Recommended addition)
Centralizes model creation logic to eliminate duplication.

### 5. Preprocessing Pipeline (`detectors/training/preprocessing/`)

#### DatasetProcessor
Centralizes dataset processing logic:
```python
from detectors.training.preprocessing import DatasetProcessorFactory

processor = DatasetProcessorFactory.create_processor('huggingface')
train_dataset, val_dataset = processor.process_split(hf_data, train_size)
```

### 6. Universal Training Script (`detectors/scripts/training/train_model.py`)

A configuration-driven training script that replaces multiple specialized scripts:
```bash
python train_model.py --config roberta_base.yaml --epochs 20 --batch_size 32
```

## Key Improvements

### 1. DRY Principle Adherence

**Before**: Dataset processing duplicated across multiple scripts
```python
# In train_roberta_new.py
def process_huggingface_dataset(hf_dataset, selection_size, human_label=0, ai_label=3):
    # 50+ lines of processing logic

# In train_ghostbuster_new.py  
def process_dataset(hf_dataset, selection_size):
    # Similar 50+ lines of processing logic
```

**After**: Centralized processing
```python
# Single processor handles all HuggingFace datasets
processor = DatasetProcessorFactory.create_processor('huggingface')
dataset = processor.process(hf_dataset, selection_size)
```

### 2. Configuration Management

**Before**: Hardcoded parameters in scripts
```python
# Scattered configuration
learning_rate = 2e-5
batch_size = 64
epochs = 10
# ... many more parameters
```

**After**: Centralized YAML configuration
```yaml
training:
  learning_rate: 2e-5
  batch_size: 64
  epochs: 10
  # All parameters in one place
```

### 3. Reduced Script Proliferation

**Before**: Multiple specialized scripts
- `train_roberta_new.py` (301 lines)
- `train_ghostbuster_new.py` (281 lines)
- `train_perplexity_new.py` (247 lines)
- Plus legacy versions...

**After**: Single universal script
- `train_model.py` (200 lines) handles all model types
- Configuration files define specifics
- Legacy scripts moved to `legacy/` folder

## Migration Guide

### For Existing Training Scripts

1. **Extract Configuration**: Move hardcoded parameters to YAML config files
2. **Use Factories**: Replace direct instantiation with factory methods
3. **Centralize Processing**: Use `DatasetProcessor` instead of custom functions
4. **Standardize Results**: Return `TrainingResult` objects

### Example Migration

**Before** (`train_roberta_new.py`):
```python
# 50+ lines of argument parsing
# 30+ lines of dataset processing
# 20+ lines of model/trainer setup
# 15+ lines of result handling
```

**After** (using new architecture):
```python
# Load configuration
config = load_config_from_file(args.config)

# Create components using factories
processor = DatasetProcessorFactory.create_processor('huggingface')
model = ModelFactory.create_model(config['model'])
trainer = TrainerFactory.create_from_config_type(model, config['training'])

# Process data
train_dataset = processor.process(hf_data, train_size)

# Train
result = trainer.train(train_dataset, val_dataset, test_dataset)
```

## Configuration Examples

### RoBERTa Training
```yaml
# roberta_base.yaml
model:
  model_name: "FacebookAI/roberta-base"
  num_labels: 2

training:
  epochs: 10
  batch_size: 64
  learning_rate: 2e-5
  
dataset:
  processor:
    human_label: 0
    ai_label: 3
```

### Ghostbuster Training
```yaml
# ghostbuster.yaml
model:
  model_name: "ghostbuster"
  
training:
  approach: "ghostbuster"
  epochs: 15
  
dataset:
  processor:
    type: "ghostbuster_specific"
```

## Best Practices

### 1. Configuration Management
- Use YAML for complex configurations
- Support command-line overrides
- Validate configurations at startup
- Save final configuration with results

### 2. Factory Usage
- Register new trainers/processors through factories
- Use type-safe enums for trainer types
- Provide clear error messages for unsupported types

### 3. Preprocessing
- Make processors stateless and reusable
- Support different data sources through processor interface
- Log processing steps for debugging

### 4. Error Handling
- Wrap Neptune/external service calls in try-catch
- Provide graceful degradation when services unavailable
- Log warnings instead of failing for non-critical issues

## Directory Structure

```
detectors/
├── training/
│   ├── configs.py              # Configuration classes
│   ├── interfaces.py           # Core interfaces
│   ├── factories/
│   │   ├── trainer_factory.py  # Trainer creation
│   │   └── model_factory.py    # Model creation
│   ├── preprocessing/
│   │   ├── dataset_processors.py
│   │   └── text_processors.py
│   └── trainers/
│       ├── base_trainer.py
│       ├── huggingface_trainer.py
│       └── ...
├── scripts/
│   ├── training/
│   │   ├── train_model.py      # Universal script
│   │   ├── configs/            # Training configurations
│   │   └── legacy/             # Moved old scripts
│   └── evaluation/
│       └── evaluate_model.py   # Universal evaluation
└── ...
```

## Future Enhancements

1. **Hyperparameter Optimization**: Integration with Optuna/Ray Tune
2. **Distributed Training**: Multi-GPU/multi-node support
3. **Model Registry**: Centralized model versioning
4. **Pipeline Integration**: CI/CD for automated training
5. **Monitoring**: Real-time training monitoring dashboard

## Migration Timeline

1. **Phase 1** ✅: Core architecture (interfaces, factories, preprocessing)
2. **Phase 2**: Migrate existing scripts to use new architecture
3. **Phase 3**: Deprecate old scripts and move to legacy
4. **Phase 4**: Add advanced features (hyperparameter optimization, etc.)

This architecture provides a solid foundation for scalable, maintainable training workflows while following industry best practices. 