# MangoTruth Dataset Registry Guide

The MangoTruth Dataset Registry provides a centralized system for discovering, loading, and managing plagiarism detection datasets. It offers pre-configured datasets with easy-to-use names and automatic setup.

## Features

- 🚀 **Easy Dataset Loading**: Load datasets with simple names like `registry.load_dataset("student-essays")`
- 🔍 **Dataset Discovery**: Browse datasets by category, size, use case, and tags
- 📊 **Multiple Sources**: Support for HuggingFace Hub, local files, and custom sources
- 🎯 **Smart Recommendations**: Get dataset suggestions based on your use case
- ⚙️ **Flexible Loading**: Control splits, sampling, shuffling, and preprocessing
- 🔄 **Dataset Merging**: Combine multiple datasets for comprehensive training

## Quick Start

### Basic Setup

```python
from detectors.data.datasets import DatasetRegistry

# Initialize the registry
registry = DatasetRegistry(cache_dir="./cache/datasets")
```

### Load a Dataset

```python
# Load by name - automatically configures and downloads
dataset = registry.load_dataset("webtext-academic")

# With options
dataset = registry.load_dataset(
    "student-essays",
    split="train",          # Specific split
    max_samples=10000,      # Limit samples
    shuffle=True,           # Shuffle data
    seed=42                 # Reproducible
)

# Explore the dataset
print(f"Dataset: {len(dataset)} samples")
print(f"Labels: {dataset.get_labels()}")
print(f"Authors: {dataset.get_authors()}")

# Access samples
sample = dataset[0]
print(f"Prompt: {sample.prompt}")
print(f"Output: {sample.output}")
print(f"Label: {sample.label}")
```

## Available Datasets

### By Category

#### Academic Datasets
- `webtext-academic`: Academic text corpus for AI vs human detection (50K samples)
- `student-essays`: Student essays vs AI-generated academic content (25K samples)
- `scientific-abstracts`: Scientific paper abstracts with AI counterparts (15K samples)

#### News & Journalism
- `news-articles`: News articles and AI-generated news content (40K samples)

#### Creative Content
- `creative-writing`: Creative writing samples from humans and AI (20K samples)

#### Technical Content
- `code-comments`: Programming code comments and AI documentation (30K samples)

#### Social Media
- `social-media`: Social media posts and AI-generated social content (100K samples)

#### Multilingual
- `multilingual-text`: Multilingual corpus for cross-language detection (75K samples)

#### Benchmark Datasets
- `hc3-english`: HC3 Human ChatGPT Comparison Corpus (80K samples)
- `ghostbuster-data`: Ghostbuster evaluation dataset (15K samples)
- `openai-detection`: OpenAI's GPT-2 detection dataset (5K samples)

### By Size

```python
# Small datasets (<20K samples) - good for experimentation
small_datasets = registry.get_datasets_by_size("small")

# Medium datasets (20K-60K samples) - balanced training
medium_datasets = registry.get_datasets_by_size("medium")

# Large datasets (>60K samples) - comprehensive training
large_datasets = registry.get_datasets_by_size("large")
```

### By Use Case

```python
# Academic research and paper writing detection
academic_datasets = registry.get_recommended_datasets("academic-research")

# Content moderation and authenticity verification
moderation_datasets = registry.get_recommended_datasets("content-moderation")

# Educational anti-cheating systems
education_datasets = registry.get_recommended_datasets("educational")

# Creative content authenticity
creative_datasets = registry.get_recommended_datasets("creative-screening")

# Technical documentation review
technical_datasets = registry.get_recommended_datasets("technical-review")

# Multilingual applications
multilingual_datasets = registry.get_recommended_datasets("cross-lingual")

# Model evaluation and benchmarking
benchmark_datasets = registry.get_recommended_datasets("benchmarking")
```

## Dataset Discovery

### Browse All Available Datasets

```python
# List all datasets
all_datasets = registry.list_available_datasets()
for name, info in all_datasets.items():
    print(f"{name}: {info.description}")
    print(f"  Size: {info.size_estimate:,} samples")
    print(f"  Tags: {info.tags}")
    print(f"  License: {info.license}")
```

### Discover by Category

```python
# Get datasets by category
academic_datasets = registry.get_datasets_by_category("academic")
benchmark_datasets = registry.get_datasets_by_category("benchmark")
multilingual_datasets = registry.get_datasets_by_category("multilingual")

# List all categories
categories = registry.list_categories()
for category, description in categories.items():
    print(f"{category}: {description}")
```

### Filter by Tags

```python
# Find datasets with specific characteristics
academic_tagged = registry.get_datasets_by_tag("academic")
benchmark_tagged = registry.get_datasets_by_tag("benchmark")
short_form_tagged = registry.get_datasets_by_tag("short-form")
```

### Preview Datasets

```python
# Get a preview without loading the full dataset
preview = registry.preview_dataset("student-essays", num_samples=3)

print(f"Dataset: {preview['dataset_name']}")
print(f"Total size: {preview['total_estimated']:,} samples")
print(f"Labels: {preview['labels']}")

for i, sample in enumerate(preview['samples']):
    print(f"Sample {i+1}:")
    print(f"  Prompt: {sample['prompt']}")
    print(f"  Output: {sample['output']}")
    print(f"  Label: {sample['label']}")
```

## Advanced Usage

### Loading Multiple Datasets

```python
# Load multiple datasets separately
datasets = registry.load_multiple_datasets(
    ["student-essays", "scientific-abstracts"],
    merge=False,
    max_samples=5000
)

for name, dataset in datasets.items():
    print(f"{name}: {len(dataset)} samples")

# Load and merge multiple datasets
merged_dataset = registry.load_multiple_datasets(
    ["webtext-academic", "student-essays"],
    merge=True,
    shuffle=True,
    seed=42
)

print(f"Merged dataset: {len(merged_dataset)} samples")
print(f"Metadata: {merged_dataset.metadata}")
```

### Custom Dataset Registration

```python
# Register your own dataset configuration
registry.register_custom_dataset(
    name="my-custom-dataset",
    description="Custom dataset for specific domain",
    source_type="huggingface",
    source_config={
        "dataset_name": "my-org/my-dataset",
        "prompt_column": "question",
        "output_column": "answer",
        "author_column": "source",
        "label_column": "is_ai",
        "label_mapping": {0: "human", 1: "ai"}
    },
    size_estimate=10000,
    splits=["train", "validation", "test"],
    tags=["custom", "domain-specific"],
    citation="My Dataset (2024)",
    license="CC-BY-4.0"
)

# Use your custom dataset
custom_dataset = registry.load_dataset("my-custom-dataset")
```

### Dataset Analysis

```python
# Get comprehensive overview
discovery = registry.discover_datasets()
print(f"Total datasets: {discovery['total_datasets']}")
print(f"Size distribution: {discovery['size_distribution']}")
print(f"Total estimated samples: {discovery['total_estimated_samples']:,}")

# Estimate download sizes
academic_names = [d.name for d in registry.get_datasets_by_category("academic")]
size_estimate = registry.estimate_download_size(academic_names)
print(f"Estimated download: {size_estimate['estimated_size_mb']:.1f} MB")

# Validate dataset configurations
validation = registry.validate_dataset_config("student-essays")
print(f"Validation: {validation}")
```

## Integration with Model Training

### Complete Training Pipeline

```python
from detectors.data.datasets import DatasetRegistry
from detectors.models.zoo import ModelRegistry, HuggingFaceNexus

# Set up registries
dataset_registry = DatasetRegistry()
model_nexus = HuggingFaceNexus()
model_registry = ModelRegistry(nexus=model_nexus)

# Load dataset for academic research
train_dataset = dataset_registry.load_dataset(
    "webtext-academic",
    split="train",
    shuffle=True,
    seed=42
)

val_dataset = dataset_registry.load_dataset(
    "webtext-academic", 
    split="validation"
)

# Load appropriate model
model = model_registry.load_pretrained_model("mango-bert-base")

# Train (simplified - actual training would use trainer)
# trainer.train(model, train_dataset, val_dataset)
```

### Recommended Combinations

| Use Case | Recommended Dataset | Recommended Model | Notes |
|----------|-------------------|------------------|-------|
| Academic Research | `webtext-academic` | `mango-bert-base` | High-quality academic content |
| Student Essays | `student-essays` | `mango-roberta-accurate` | Educational focus |
| Content Moderation | `news-articles` | `mango-distilbert-fast` | Real-time processing |
| Creative Screening | `creative-writing` | `mango-electra-efficient` | Artistic content |
| Technical Review | `code-comments` | `mango-bert-base` | Technical language |
| Multilingual | `multilingual-text` | `mango-deberta-multilingual` | Cross-language support |
| Benchmarking | `hc3-english` | `mango-roberta-accurate` | Standard evaluation |

## Dataset Processing

### Filtering and Transformation

```python
# Load dataset
dataset = registry.load_dataset("student-essays")

# Filter by label
human_samples = dataset.get_by_label("human")
ai_samples = dataset.get_by_label("ai")

# Filter by author
specific_author = dataset.get_by_author("gpt-4")

# Custom filtering
long_texts = dataset.filter(lambda sample: len(sample.output) > 1000)

# Apply transformations
def preprocess_sample(sample):
    # Custom preprocessing
    sample.output = sample.output.lower()
    return sample

processed_dataset = dataset.map(preprocess_sample)
```

### Dataset Splitting

```python
# Split dataset for training
train_ds, val_ds, test_ds = dataset.split([0.7, 0.15, 0.15], seed=42)

print(f"Train: {len(train_ds)} samples")
print(f"Validation: {len(val_ds)} samples") 
print(f"Test: {len(test_ds)} samples")
```

### Batch Processing

```python
# Process in batches
batch_size = 32
for batch in dataset.batch(batch_size):
    # Process batch of samples
    texts = [sample.output for sample in batch]
    labels = [sample.label for sample in batch]
    # ... batch processing logic
```

## Best Practices

### Choosing Datasets

1. **Academic Research**: Use `webtext-academic` or `student-essays` for formal writing
2. **Production Systems**: Use `news-articles` or `social-media` for diverse content
3. **Benchmarking**: Use `hc3-english` or `ghostbuster-data` for standardized evaluation
4. **Experimentation**: Start with smaller datasets like `scientific-abstracts`
5. **Multilingual**: Use `multilingual-text` for cross-language applications

### Performance Optimization

```python
# For large datasets, use sampling
large_dataset = registry.load_dataset(
    "social-media",
    max_samples=50000,  # Limit size
    shuffle=True,       # Randomize selection
    seed=42            # Reproducible
)

# For memory efficiency, process in batches
for batch in large_dataset.batch(1000):
    # Process batch instead of all at once
    pass
```

### Data Quality

```python
# Validate dataset before use
validation = registry.validate_dataset_config("my-dataset")
if not validation["valid"]:
    print(f"Validation failed: {validation}")

# Preview dataset structure
preview = registry.preview_dataset("my-dataset", num_samples=5)
print(f"Sample structure: {preview}")

# Check dataset balance
dataset = registry.load_dataset("my-dataset")
labels = dataset.get_labels()
for label in labels:
    count = len(dataset.get_by_label(label))
    print(f"{label}: {count} samples")
```

## Configuration Reference

### DatasetInfo Structure

```python
@dataclass
class DatasetInfo:
    name: str                    # Unique dataset name
    description: str             # Human-readable description
    source_type: str            # "huggingface", "local", "url"
    source_config: Dict         # Configuration for data source
    size_estimate: int          # Estimated number of samples
    splits: List[str]           # Available splits
    tags: List[str]             # Tags for categorization
    citation: str               # Citation information
    license: str                # License information
```

### HuggingFace Source Config

```python
source_config = {
    "dataset_name": "my-org/my-dataset",     # HuggingFace dataset ID
    "config": "subset_name",                 # Optional dataset config
    "prompt_column": "question",             # Column containing prompts
    "output_column": "answer",               # Column containing outputs
    "author_column": "source",               # Column containing author info
    "label_column": "is_ai",                 # Column containing labels
    "label_mapping": {0: "human", 1: "ai"}  # Map numeric to string labels
}
```

## Error Handling

```python
try:
    dataset = registry.load_dataset("non-existent-dataset")
except ValueError as e:
    print(f"Dataset not found: {e}")
    # Fall back to available dataset
    available = registry.list_available_datasets()
    print(f"Available datasets: {list(available.keys())}")

try:
    # Load with preview first for large datasets
    preview = registry.preview_dataset("large-dataset")
    if preview.get("error"):
        print(f"Dataset preview failed: {preview['error']}")
    else:
        dataset = registry.load_dataset("large-dataset", max_samples=1000)
except Exception as e:
    print(f"Loading failed: {e}")
```

## Contributing

To add new datasets to the registry:

1. Add dataset configuration to `default_datasets.py`
2. Include in appropriate categories
3. Add performance metrics and metadata
4. Upload dataset to HuggingFace Hub (if applicable)
5. Update documentation and examples

## License

This dataset registry is part of the MangoTruth plagiarism detection framework. Individual datasets have their own licenses as specified in their metadata. 