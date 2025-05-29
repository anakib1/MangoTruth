# Evaluation Architecture Documentation

This document outlines the centralized evaluation architecture that provides clear separation between training and evaluation while eliminating code duplication.

## Overview

The evaluation system has been designed to provide a **single source of truth** for model evaluation that can be used both during training and for standalone evaluation, eliminating the duplication that previously existed across multiple modules.

## Architecture Principles

### 1. **Single Responsibility**
- **Training**: Focuses solely on optimization and model parameter updates
- **Evaluation**: Handles model assessment, metrics calculation, and result analysis

### 2. **Centralized Logic**
- All evaluation logic consolidated in `detectors/evaluation/`
- Single `StandardEvaluator` used by both trainers and standalone scripts
- Unified metrics calculation through `MetricsCalculator`

### 3. **Consistent Interface**
- Same evaluation API used during training validation and standalone evaluation
- Standardized `EvaluationResult` format across all evaluation contexts
- Backward compatibility maintained for existing trainer interfaces

## Core Components

### 1. Evaluation Interfaces (`detectors/evaluation/interfaces.py`)

```python
@dataclass
class EvaluationResult:
    """Unified result format for all evaluation operations."""
    conclusion: SplitConclusion          # Metrics and visualizations
    predictions: np.ndarray              # Raw model predictions  
    true_labels: np.ndarray              # Ground truth labels
    dataset_info: Dict[str, Any]         # Dataset metadata
    model_info: Dict[str, Any]           # Model metadata
    evaluation_config: Dict[str, Any]    # Evaluation parameters

class IEvaluator(ABC):
    """Interface for model evaluators."""
    def evaluate(self, model, dataset, **kwargs) -> EvaluationResult
    def batch_evaluate(self, model, datasets, **kwargs) -> Dict[str, EvaluationResult]
```

### 2. Centralized Metrics (`detectors/evaluation/metrics.py`)

**Before**: Metrics logic scattered across:
- `detectors/utils/training.py` 
- `detectors/models/zoo/evaluation.py`
- Individual trainer implementations

**After**: Single `MetricsCalculator` with:
```python
class MetricsCalculator(IMetricsCalculator):
    def calculate_metrics(self, true_labels, predictions, **kwargs) -> SplitConclusion
    def calculate_batch_metrics(self, results, **kwargs) -> Dict[str, SplitConclusion]
```

### 3. Unified Evaluator (`detectors/evaluation/evaluators.py`)

**Consolidates logic from**:
- `ModelEvaluator` (zoo/evaluation.py) → `StandardEvaluator`
- Trainer evaluation methods → Use `StandardEvaluator`
- Training utility functions → Use `MetricsCalculator`

## Clear Training/Evaluation Separation

### Training Side (`detectors/training/`)

**Responsibilities**:
- Model parameter optimization
- Training loop management
- Checkpoint saving/loading
- Learning rate scheduling

**Evaluation Usage**:
```python
class BaseTrainer(ITrainer):
    def __init__(self, model, config):
        self._evaluator = StandardEvaluator()  # Use centralized evaluator
    
    def evaluate(self, dataset, **kwargs) -> Conclusion:
        # Delegate to centralized evaluator
        result = self._evaluator.evaluate(self.model, dataset, **kwargs)
        return result.conclusion
```

### Evaluation Side (`detectors/evaluation/`)

**Responsibilities**:
- Model performance assessment
- Metrics calculation and visualization  
- Multi-dataset evaluation
- Result analysis and reporting

**Standalone Usage**:
```python
# Completely independent from training
evaluator = StandardEvaluator()
result = evaluator.evaluate(model, dataset)
```

## Eliminated Duplication

### 1. **Dataset Label Mapping**

**Before**: Duplicated across trainers and evaluators
```python
# In HuggingFaceTrainer._create_label_mapping()
# In ModelEvaluator.evaluate_dataset()
# Similar logic repeated 3+ times
```

**After**: Single implementation in `StandardEvaluator._create_label_mapping()`

### 2. **Prediction Processing**

**Before**: Different prediction handling in each trainer/evaluator
```python
# In HuggingFaceTrainer
# In ModelEvaluator._prepare_predictions()
# In various training scripts
```

**After**: Centralized in `StandardEvaluator._prepare_predictions()`

### 3. **Metrics Calculation**

**Before**: Scattered across multiple files
```python
# detectors/utils/training.py: calculate_classification()
# zoo/evaluation.py: Various metric calculations
# Trainer-specific metric handling
```

**After**: Single `MetricsCalculator.calculate_metrics()`

### 4. **Binary/Multi-class Conversion**

**Before**: Ad-hoc conversion logic in multiple places

**After**: Centralized in `StandardEvaluator._convert_to_binary_classification()`

## Usage Examples

### 1. **During Training**

```python
# In any trainer
class MyTrainer(BaseTrainer):
    def train(self, train_dataset, val_dataset, test_dataset):
        # Training loop...
        
        # Validation (uses centralized evaluation)
        val_result = self.evaluate_comprehensive(val_dataset)
        
        # Test evaluation (same evaluator)
        test_result = self.evaluate_comprehensive(test_dataset)
```

### 2. **Standalone Evaluation**

```python
# Completely separate from training
python evaluate_model.py \
    --model_path ./models/roberta_detector \
    --dataset_handle anakib1/mango-truth \
    --dataset_name xlsum \
    --batch_size 32 \
    --save_predictions
```

### 3. **Programmatic Evaluation**

```python
from detectors.evaluation import StandardEvaluator

evaluator = StandardEvaluator()

# Single dataset
result = evaluator.evaluate(model, dataset)
print(f"Accuracy: {result.conclusion.metrics.accuracy:.4f}")

# Multiple datasets
results = evaluator.batch_evaluate(model, {
    "test1": dataset1,
    "test2": dataset2
})
```

## Interface Compatibility

### Backward Compatibility

The new architecture maintains backward compatibility:

```python
# Old interface still works
trainer = HuggingFaceTrainer(model, config)
conclusion = trainer.evaluate(dataset)  # Returns Conclusion

# New interface provides more data
evaluation_result = trainer.evaluate_comprehensive(dataset)  # Returns EvaluationResult
```

### Consistent Return Types

- **Training context**: `evaluate()` returns `Conclusion` for compatibility
- **Comprehensive context**: `evaluate_comprehensive()` returns `EvaluationResult`
- **Standalone context**: Always returns `EvaluationResult`

## Benefits of This Architecture

### 1. **Zero Duplication**
- Single implementation of evaluation logic
- Shared across training and standalone evaluation
- Consistent results regardless of context

### 2. **Clear Separation of Concerns**
- Training focused on optimization
- Evaluation focused on assessment
- No mixed responsibilities

### 3. **Easier Maintenance**
- Single place to update evaluation logic
- Consistent bug fixes across all usage contexts
- Unified testing strategy

### 4. **Enhanced Functionality**
- Rich metadata in evaluation results
- Comprehensive multi-dataset evaluation
- Flexible metrics calculation

### 5. **Better Testing**
- Evaluation logic can be tested independently
- Mock evaluation for training tests
- Consistent evaluation behavior

## Directory Structure

```
detectors/
├── evaluation/                    # 🆕 Centralized evaluation
│   ├── interfaces.py             # Common interfaces and result types
│   ├── evaluators.py             # StandardEvaluator implementation  
│   ├── metrics.py                # Centralized metrics calculation
│   └── EVALUATION_ARCHITECTURE.md
├── training/
│   └── trainers/
│       └── base_trainer.py       # ✅ Uses centralized evaluation
├── scripts/
│   ├── training/
│   │   └── train_model.py        # Training-focused
│   └── evaluation/               # 🆕 Evaluation-focused
│       └── evaluate_model.py     # Standalone evaluation
├── models/zoo/
│   └── evaluation.py             # 🔄 Will be deprecated
└── utils/
    └── training.py               # 🔄 Still used by metrics calculator
```

## Migration Status

### ✅ **Completed**
- Centralized evaluation interfaces and implementations
- Updated `BaseTrainer` to use centralized evaluation
- Created standalone evaluation script
- Eliminated duplication in evaluation logic

### 🔄 **In Progress**  
- Update specific trainers (HuggingFace, Perplexity, Ghostbuster)
- Deprecate old `ModelEvaluator` in zoo/evaluation.py
- Update training scripts to use new evaluation

### 📋 **Future**
- Advanced evaluation features (confidence intervals, statistical tests)
- Integration with experiment tracking (Neptune, MLflow)
- Automated evaluation pipelines

## Summary

The new evaluation architecture provides **complete separation** between training and evaluation while **eliminating all duplication**. Both training and standalone evaluation now use the exact same evaluation logic, ensuring consistency and maintainability.

**Key Achievement**: A single `StandardEvaluator` instance can handle evaluation in any context - during training validation, post-training assessment, or completely standalone model evaluation - with identical results and behavior. 