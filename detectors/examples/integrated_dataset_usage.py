"""Example usage of the integrated DatasetRegistry system.

This script demonstrates how to:
1. Set up the DatasetRegistry for easy dataset loading
2. Load pre-defined datasets using simple names
3. Discover and browse available datasets
4. Work with different dataset categories and use cases
5. Register custom datasets
6. Preview and validate datasets
"""

import logging
from typing import List

from detectors.data.datasets import (
    DatasetRegistry,
    DatasetInfo,
    get_dataset_info,
    get_datasets_by_category,
    get_recommended_datasets,
    list_available_datasets
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataset_registry():
    """Set up the DatasetRegistry."""
    logger.info("Setting up DatasetRegistry...")
    
    # Initialize registry with custom cache directory
    registry = DatasetRegistry(cache_dir="./cache/datasets")
    
    return registry


def demonstrate_dataset_discovery(registry: DatasetRegistry):
    """Demonstrate dataset discovery and browsing capabilities."""
    logger.info("=== Dataset Discovery ===")
    
    # List all available datasets
    all_datasets = registry.list_available_datasets()
    logger.info(f"Total available datasets: {len(all_datasets)}")
    
    # Show some example datasets
    for i, (name, info) in enumerate(list(all_datasets.items())[:3]):
        logger.info(f"  {i+1}. {name}: {info.description}")
        logger.info(f"     Size: {info.size_estimate} samples, Tags: {info.tags}")
    
    # Browse by category
    logger.info("\n=== Datasets by Category ===")
    academic_datasets = registry.get_datasets_by_category("academic")
    logger.info(f"Academic datasets: {[d.name for d in academic_datasets]}")
    
    benchmark_datasets = registry.get_datasets_by_category("benchmark")
    logger.info(f"Benchmark datasets: {[d.name for d in benchmark_datasets]}")
    
    # Browse by size
    logger.info("\n=== Datasets by Size ===")
    small_datasets = registry.get_datasets_by_size("small")
    logger.info(f"Small datasets (<20K samples): {[d.name for d in small_datasets]}")
    
    large_datasets = registry.get_datasets_by_size("large")
    logger.info(f"Large datasets (>60K samples): {[d.name for d in large_datasets]}")
    
    # Get recommendations for use cases
    logger.info("\n=== Recommendations by Use Case ===")
    research_datasets = registry.get_recommended_datasets("academic-research")
    logger.info(f"Academic research: {[d.name for d in research_datasets]}")
    
    education_datasets = registry.get_recommended_datasets("educational")
    logger.info(f"Educational: {[d.name for d in education_datasets]}")


def demonstrate_dataset_loading(registry: DatasetRegistry):
    """Demonstrate loading datasets with different configurations."""
    logger.info("=== Dataset Loading ===")
    
    try:
        # Load a small dataset for demonstration
        logger.info("Loading scientific abstracts dataset...")
        dataset = registry.load_dataset(
            "scientific-abstracts",
            max_samples=100,  # Limit for demo
            shuffle=True,
            seed=42
        )
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        logger.info(f"Labels: {dataset.get_labels()}")
        logger.info(f"Authors: {dataset.get_authors()}")
        
        # Show a sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"\nSample:")
            logger.info(f"  Prompt: {sample.prompt[:100]}...")
            logger.info(f"  Output: {sample.output[:200]}...")
            logger.info(f"  Label: {sample.label}")
        
        return dataset
        
    except Exception as e:
        logger.warning(f"Could not load dataset (expected in demo): {e}")
        
        # Create a mock dataset instead
        from detectors.data.datasets import Dataset, TextSample
        mock_samples = [
            TextSample(
                prompt="Research topic: AI ethics",
                output="Artificial intelligence ethics is a crucial field...",
                author_id="researcher_1",
                label="human"
            ),
            TextSample(
                prompt="Research topic: AI ethics", 
                output="The ethical implications of artificial intelligence systems...",
                author_id="ai_model",
                label="ai"
            )
        ]
        return Dataset.from_samples(mock_samples)


def demonstrate_dataset_preview(registry: DatasetRegistry):
    """Demonstrate dataset preview functionality."""
    logger.info("=== Dataset Preview ===")
    
    # Preview different datasets
    datasets_to_preview = ["student-essays", "news-articles", "code-comments"]
    
    for dataset_name in datasets_to_preview:
        logger.info(f"\nPreviewing {dataset_name}:")
        preview = registry.preview_dataset(dataset_name, num_samples=2)
        
        if "error" in preview:
            logger.warning(f"  Could not preview: {preview['error']}")
            continue
        
        logger.info(f"  Description: {registry.get_dataset_info(dataset_name).description}")
        logger.info(f"  Estimated size: {preview['total_estimated']} samples")
        logger.info(f"  Labels: {preview['labels']}")
        
        for i, sample in enumerate(preview['samples']):
            logger.info(f"    Sample {i+1}:")
            logger.info(f"      Prompt: {sample['prompt']}")
            logger.info(f"      Output: {sample['output']}")
            logger.info(f"      Label: {sample['label']}")


def demonstrate_multiple_dataset_loading(registry: DatasetRegistry):
    """Demonstrate loading and merging multiple datasets."""
    logger.info("=== Multiple Dataset Loading ===")
    
    # Load multiple datasets separately
    dataset_names = ["scientific-abstracts", "code-comments"]
    
    try:
        logger.info("Loading multiple datasets separately...")
        datasets = registry.load_multiple_datasets(
            dataset_names,
            merge=False,
            max_samples=50  # Limit each dataset
        )
        
        for name, dataset in datasets.items():
            logger.info(f"  {name}: {len(dataset)} samples, labels: {dataset.get_labels()}")
        
        # Now merge them
        logger.info("Loading and merging datasets...")
        merged_dataset = registry.load_multiple_datasets(
            dataset_names,
            merge=True,
            max_samples=50
        )
        
        logger.info(f"Merged dataset: {len(merged_dataset)} samples")
        logger.info(f"Combined labels: {merged_dataset.get_labels()}")
        logger.info(f"Metadata: {merged_dataset.metadata}")
        
    except Exception as e:
        logger.warning(f"Could not load multiple datasets: {e}")


def demonstrate_custom_dataset_registration(registry: DatasetRegistry):
    """Demonstrate registering custom datasets."""
    logger.info("=== Custom Dataset Registration ===")
    
    # Register a custom dataset
    registry.register_custom_dataset(
        name="my-custom-dataset",
        description="Custom dataset for specific domain detection",
        source_type="huggingface",
        source_config={
            "dataset_name": "my-org/my-custom-dataset",
            "prompt_column": "question",
            "output_column": "answer", 
            "author_column": "source",
            "label_column": "is_synthetic",
            "label_mapping": {0: "human", 1: "ai"}
        },
        size_estimate=10000,
        splits=["train", "validation", "test"],
        tags=["custom", "domain-specific", "experimental"],
        citation="My Custom Dataset (2024)",
        license="CC-BY-4.0"
    )
    
    # Show updated registry
    all_datasets = registry.list_available_datasets()
    logger.info(f"Total datasets after registration: {len(all_datasets)}")
    
    # Validate the custom dataset
    validation = registry.validate_dataset_config("my-custom-dataset")
    logger.info(f"Custom dataset validation: {validation}")


def demonstrate_dataset_analysis(registry: DatasetRegistry):
    """Demonstrate dataset analysis and size estimation."""
    logger.info("=== Dataset Analysis ===")
    
    # Show dataset discovery overview
    discovery = registry.discover_datasets()
    logger.info(f"Dataset discovery overview:")
    logger.info(f"  Total datasets: {discovery['total_datasets']}")
    logger.info(f"  Predefined: {discovery['predefined']}")
    logger.info(f"  Custom: {discovery['custom']}")
    logger.info(f"  Size distribution: {discovery['size_distribution']}")
    logger.info(f"  Estimated total samples: {discovery['total_estimated_samples']:,}")
    
    # Show categories
    logger.info(f"\nAvailable categories:")
    categories = registry.list_categories()
    for cat, desc in categories.items():
        count = len(registry.get_datasets_by_category(cat))
        logger.info(f"  {cat}: {desc} ({count} datasets)")
    
    # Show use cases
    logger.info(f"\nAvailable use cases:")
    use_cases = registry.list_use_cases()
    for case, desc in use_cases.items():
        count = len(registry.get_recommended_datasets(case))
        logger.info(f"  {case}: {desc} ({count} datasets)")
    
    # Estimate download size
    academic_datasets = [d.name for d in registry.get_datasets_by_category("academic")]
    size_estimate = registry.estimate_download_size(academic_datasets[:3])  # First 3
    logger.info(f"\nDownload size estimate for academic datasets:")
    logger.info(f"  Datasets: {size_estimate['datasets']}")
    logger.info(f"  Estimated samples: {size_estimate['estimated_samples']:,}")
    logger.info(f"  Estimated size: {size_estimate['estimated_size_mb']:.1f} MB")


def demonstrate_dataset_filtering(registry: DatasetRegistry):
    """Demonstrate filtering and searching datasets."""
    logger.info("=== Dataset Filtering ===")
    
    # Filter by tags
    academic_tagged = registry.get_datasets_by_tag("academic")
    logger.info(f"Datasets with 'academic' tag: {[d.name for d in academic_tagged]}")
    
    benchmark_tagged = registry.get_datasets_by_tag("benchmark")
    logger.info(f"Datasets with 'benchmark' tag: {[d.name for d in benchmark_tagged]}")
    
    multilingual_tagged = registry.get_datasets_by_tag("multilingual")
    logger.info(f"Datasets with 'multilingual' tag: {[d.name for d in multilingual_tagged]}")
    
    # Show dataset details for interesting ones
    logger.info(f"\nDetailed information for selected datasets:")
    for dataset_name in ["webtext-academic", "hc3-english", "multilingual-text"]:
        info = registry.get_dataset_info(dataset_name)
        if info:
            logger.info(f"  {dataset_name}:")
            logger.info(f"    Description: {info.description}")
            logger.info(f"    Size: {info.size_estimate:,} samples")
            logger.info(f"    Splits: {info.splits}")
            logger.info(f"    Tags: {info.tags}")
            logger.info(f"    License: {info.license}")


def demonstrate_integration_with_models():
    """Demonstrate how datasets integrate with the model system."""
    logger.info("=== Dataset + Model Integration ===")
    
    # This shows how datasets can be used with the model registry
    from detectors.models.zoo import ModelRegistry, HuggingFaceNexus
    
    # Set up integrated system
    dataset_registry = DatasetRegistry()
    model_nexus = HuggingFaceNexus()
    model_registry = ModelRegistry(nexus=model_nexus)
    
    logger.info("Integrated system setup complete")
    
    # Recommend model-dataset combinations
    logger.info("\nRecommended model-dataset combinations:")
    
    combinations = [
        ("academic-research", "mango-bert-base"),
        ("content-moderation", "mango-distilbert-fast"),
        ("benchmarking", "mango-roberta-accurate")
    ]
    
    for use_case, model_name in combinations:
        datasets = dataset_registry.get_recommended_datasets(use_case)
        dataset_names = [d.name for d in datasets[:2]]  # First 2
        
        logger.info(f"  {use_case.title()}:")
        logger.info(f"    Model: {model_name}")
        logger.info(f"    Datasets: {dataset_names}")
        
        # Get size estimate
        if dataset_names:
            size_est = dataset_registry.estimate_download_size(dataset_names)
            logger.info(f"    Estimated data size: {size_est['estimated_size_mb']:.1f} MB")


def main():
    """Main demonstration function."""
    logger.info("Starting MangoTruth DatasetRegistry Demo")
    
    # Set up the registry
    registry = setup_dataset_registry()
    
    # Demonstrate various capabilities
    demonstrate_dataset_discovery(registry)
    demonstrate_dataset_preview(registry)
    demonstrate_dataset_loading(registry)
    demonstrate_multiple_dataset_loading(registry)
    demonstrate_custom_dataset_registration(registry)
    demonstrate_dataset_analysis(registry)
    demonstrate_dataset_filtering(registry)
    demonstrate_integration_with_models()
    
    # Show final summary
    logger.info("\n=== Final Summary ===")
    discovery = registry.discover_datasets()
    logger.info(f"DatasetRegistry now contains {discovery['total_datasets']} datasets")
    logger.info(f"Categories available: {list(discovery['categories'].keys())}")
    logger.info(f"Use cases supported: {list(discovery['use_cases'].keys())}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() 