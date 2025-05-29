# Training Pipeline for Plagiarism Detection Framework

This module provides a flexible and extensible training infrastructure for the plagiarism detection framework. It supports both deep learning models (currently focused on HuggingFace transformers) and is designed to accommodate non-deep-learning approaches in the future.

## Architecture Overview

The training pipeline follows a modular design with clear separation of concerns:

```
training/
├── interfaces.py          # Core interfaces (ITrainable, ITrainer, etc.)
├── configs.py            # Configuration classes
├── trainers/
│   ├── base_trainer.py   # Base trainer with common functionality
│   └── huggingface_trainer.py  # HuggingFace-specific trainer
└── examples/             # Example training scripts
```

## Key Components

### 1. Interfaces

#### ITrainable
The `ITrainable` interface extends the base `IDetector` interface with training capabilities:

```python
from detectors.training.interfaces import ITrainable

class MyTrainableModel(IDetector, ITrainable):
    def train_mode(self, mode: bool = True) -> None:
        # Switch between training and evaluation mode
        pass
    
    def compute_loss(self, inputs, labels):
        # Compute loss for training
        pass
    
    # ... other required methods
```

#### ITrainer
The `ITrainer` interface defines the contract for trainer implementations:

```python
from detectors.training.interfaces import ITrainer

class MyTrainer(ITrainer):
    def train(self, train_dataset, validation_dataset=None):
        # Implement training logic
        pass
    
    def evaluate(self, dataset):
        # Evaluate model on dataset
        pass
```

### 2. Configuration

Training configurations are hierarchical, with `TrainingConfig` as the base:

```python
from detectors.training.configs import HuggingFaceTrainingConfig

config = HuggingFaceTrainingConfig(
    model_name="bert-base-uncased",
    num_labels=2,
    epochs=3,
    batch_size=16,
    learning_rate=5e-5,
    early_stopping=True,
    output_dir="./output"
)
```

### 3. Trainers

#### HuggingFace Trainer
The `HuggingFaceTrainer` leverages the HuggingFace transformers library:

```python
from detectors.training import HuggingFaceTrainer
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector

# Create model
model = TrainableHuggingFaceDetector(model_config)

# Create trainer
trainer = HuggingFaceTrainer(model, training_config)

# Train
result = trainer.train(train_dataset, validation_dataset)
```

## Usage Example

Here's a complete example of training a plagiarism detector:

```python
from detectors.data.datasets import Dataset, TextSample
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.training import HuggingFaceTrainer, HuggingFaceTrainingConfig

# 1. Prepare your dataset
samples = [
    TextSample(
        prompt="Write about AI",
        output="Artificial intelligence is...",
        author_id="human",
        label="human"
    ),
    TextSample(
        prompt="Write about AI",
        output="AI represents a paradigm shift...",
        author_id="gpt-4",
        label="ai"
    ),
    # ... more samples
]
dataset = Dataset.from_samples(samples)
train_dataset, val_dataset = dataset.split([0.8, 0.2])

# 2. Configure the model
model_config = HuggingFaceConfig(
    model_name="distilbert-base-uncased",
    num_labels=2,
    max_length=256
)

# 3. Initialize trainable model
model = TrainableHuggingFaceDetector(model_config)

# 4. Configure training
training_config = HuggingFaceTrainingConfig(
    model_name="distilbert-base-uncased",
    epochs=3,
    batch_size=16,
    learning_rate=5e-5,
    output_dir="./training_output"
)

# 5. Create trainer and train
trainer = HuggingFaceTrainer(model, training_config)
result = trainer.train(train_dataset, val_dataset)

# 6. Save the trained model
trainer.save_model("./models/my_detector")
```

## Extending the Framework

### Adding a New Deep Learning Trainer

To add support for a different deep learning framework (e.g., PyTorch Lightning):

1. Create a new trainable model class:
```python
class TrainablePyTorchModel(YourBaseModel, ITrainable):
    # Implement ITrainable methods
    pass
```

2. Create a new trainer:
```python
from detectors.training.trainers.base_trainer import BaseTrainer

class PyTorchLightningTrainer(BaseTrainer):
    def train(self, train_dataset, validation_dataset=None):
        # Implement PyTorch Lightning training
        pass
```

### Adding Non-Deep Learning Models

The framework is designed to support classical ML approaches:

1. Create a trainable model:
```python
class TrainableSVMDetector(IDetector, ITrainable):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.svm = SVC(probability=True)
    
    def train_mode(self, mode: bool = True):
        # SVMs don't have train/eval modes
        pass
    
    def compute_loss(self, inputs, labels):
        # Compute hinge loss or similar
        pass
```

2. Create a classical ML trainer:
```python
class ClassicalMLTrainer(BaseTrainer):
    def train(self, train_dataset, validation_dataset=None):
        # Extract features
        # Train model
        # Evaluate
        pass
```

## Training Results

The `TrainingResult` object contains comprehensive information about the training run:

```python
result = trainer.train(train_dataset, val_dataset)

# Access metrics
print(f"Training duration: {result.training_duration_seconds}s")
print(f"Best checkpoint: {result.best_checkpoint_path}")
print(f"Final accuracy: {result.validation_conclusion.metrics.accuracy}")

# Access training history
for epoch, loss in enumerate(result.training_history['loss']):
    print(f"Epoch {epoch}: Loss = {loss}")
```

## Best Practices

1. **Dataset Preparation**: Ensure your dataset has balanced classes and sufficient examples.

2. **Configuration**: Start with default configurations and adjust based on your needs:
   - Smaller models (DistilBERT) for quick experiments
   - Larger models (BERT, RoBERTa) for better performance

3. **Monitoring**: The trainer logs progress and saves checkpoints automatically.

4. **Early Stopping**: Enable early stopping to prevent overfitting:
   ```python
   config = HuggingFaceTrainingConfig(
       early_stopping=True,
       early_stopping_patience=3
   )
   ```

5. **GPU Usage**: The framework automatically detects and uses available GPUs.

## Integration with Existing Code

The training pipeline integrates seamlessly with the existing detector framework:

- Trained models implement the same `IDetector` interface
- Models can be saved/loaded using the standard methods
- Evaluation uses the existing `ModelEvaluator` class

## Future Enhancements

The framework is designed to support:

- Distributed training
- Hyperparameter optimization
- AutoML approaches
- Online/incremental learning
- Multi-task learning
- Ensemble methods

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision training (fp16/bf16)
3. **Poor Performance**: Check dataset quality and try different models

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
``` 