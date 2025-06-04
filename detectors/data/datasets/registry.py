"""Dataset registry for managing and loading pre-configured datasets.

This module provides a DatasetRegistry that enables easy loading of pre-defined
datasets by name, with support for different sources and automatic configuration.
"""

from typing import Dict, Optional, List, Any, Union
import logging

from detectors.data.datasets.dataset import Dataset
from detectors.data.datasets.loaders.huggingface_loader import HuggingFaceLoader  
from detectors.data.datasets.factory import create_dataset
from detectors.data.datasets.default_datasets import (
    DatasetInfo,
    DEFAULT_HUGGINGFACE_DATASETS,
    BENCHMARK_DATASETS,
    DATASET_CATEGORIES,
    USAGE_RECOMMENDATIONS,
    get_dataset_info,
    list_available_datasets,
    get_datasets_by_category,
    get_datasets_by_tag,
    get_recommended_datasets,
    get_datasets_by_size,
    estimate_total_size
)

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for managing and loading pre-configured datasets.
    
    This registry provides:
    - Easy loading of pre-defined dataset presets by name
    - Dataset discovery and browsing by category and use case
    - Support for multiple data sources (HuggingFace, local files, etc.)
    - Automatic configuration of loaders and backends
    - Dataset metadata and documentation
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the dataset registry.
        
        Args:
            cache_dir: Optional cache directory for downloaded datasets
        """
        self.cache_dir = cache_dir or "./cache/datasets"
        self._custom_datasets: Dict[str, DatasetInfo] = {}

    def load_dataset(self, 
                    dataset_name: str, 
                    split: Optional[Union[str, List[str]]] = None,
                    max_samples: Optional[int] = None,
                    shuffle: bool = False,
                    seed: Optional[int] = None,
                    **kwargs) -> Dataset:
        """Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Specific split(s) to load (defaults to all available)
            max_samples: Maximum number of samples to load
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling/sampling
            **kwargs: Additional arguments for the loader
            
        Returns:
            Loaded Dataset instance
            
        Raises:
            ValueError: If dataset is not found or loading fails
        """
        # Get dataset info
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found. "
                           f"Available datasets: {list(self.list_available_datasets().keys())}")
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Create the appropriate loader
        if dataset_info.source_type == "huggingface":
            dataset = self._load_huggingface_dataset(dataset_info, split, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {dataset_info.source_type}")
        
        # Apply post-processing
        if max_samples and len(dataset) > max_samples:
            if shuffle or seed is not None:
                dataset = dataset.shuffle(seed=seed)
            # Take first max_samples
            indices = list(range(min(max_samples, len(dataset))))
            dataset = dataset[indices]
            logger.info(f"Limited dataset to {max_samples} samples")
        elif shuffle:
            dataset = dataset.shuffle(seed=seed)
            logger.info("Shuffled dataset")
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        logger.info(f"Labels: {dataset.get_labels()}")
        
        return dataset

    def _load_huggingface_dataset(self, 
                                 dataset_info: DatasetInfo, 
                                 split: Optional[Union[str, List[str]]] = None,
                                 **kwargs) -> Dataset:
        """Load a HuggingFace dataset using the configured loader."""
        source_config = dataset_info.source_config.copy()
        source_config.update(kwargs)
        
        # Override split if specified
        if split is not None:
            source_config["split"] = split
        
        # Create loader
        loader = HuggingFaceLoader(**source_config)
        
        # Create dataset with loader
        dataset = create_dataset("memory")
        loader.load(dataset.backend)
        
        # Store metadata
        dataset.metadata.update({
            "dataset_name": dataset_info.name,
            "description": dataset_info.description,
            "source_type": dataset_info.source_type,
            "citation": dataset_info.citation,
            "license": dataset_info.license,
            "tags": dataset_info.tags
        })
        
        return dataset

    def load_multiple_datasets(self, 
                              dataset_names: List[str],
                              merge: bool = True,
                              **kwargs) -> Union[Dataset, Dict[str, Dataset]]:
        """Load multiple datasets.
        
        Args:
            dataset_names: List of dataset names to load
            merge: Whether to merge datasets into one or return separately
            **kwargs: Additional arguments for loading
            
        Returns:
            Single merged Dataset or dictionary of datasets
        """
        datasets = {}
        
        for name in dataset_names:
            logger.info(f"Loading dataset: {name}")
            datasets[name] = self.load_dataset(name, **kwargs)
        
        if merge:
            logger.info("Merging datasets...")
            from detectors.data.datasets.factory import merge_datasets
            merged = merge_datasets(list(datasets.values()))
            
            # Update metadata
            merged.metadata["dataset_names"] = dataset_names
            merged.metadata["merged"] = True
            merged.metadata["total_samples"] = len(merged)
            
            return merged
        
        return datasets

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information or None if not found
        """
        # Check custom datasets first
        if dataset_name in self._custom_datasets:
            return self._custom_datasets[dataset_name]
        
        # Check predefined datasets
        return get_dataset_info(dataset_name)

    def list_available_datasets(self) -> Dict[str, DatasetInfo]:
        """List all available datasets.
        
        Returns:
            Dictionary mapping dataset names to dataset information
        """
        datasets = list_available_datasets()
        datasets.update(self._custom_datasets)
        return datasets

    def get_datasets_by_category(self, category: str) -> List[DatasetInfo]:
        """Get datasets by category.
        
        Args:
            category: Category name (e.g., 'academic', 'news', 'benchmark')
            
        Returns:
            List of dataset information for the category
        """
        return get_datasets_by_category(category)

    def get_datasets_by_tag(self, tag: str) -> List[DatasetInfo]:
        """Get datasets by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of dataset information with the specified tag
        """
        all_datasets = self.list_available_datasets()
        return [dataset for dataset in all_datasets.values() if tag in dataset.tags]

    def get_recommended_datasets(self, use_case: str) -> List[DatasetInfo]:
        """Get recommended datasets for a specific use case.
        
        Args:
            use_case: Use case name (e.g., 'academic-research', 'content-moderation')
            
        Returns:
            List of recommended dataset information
        """
        return get_recommended_datasets(use_case)

    def get_datasets_by_size(self, size_category: str) -> List[DatasetInfo]:
        """Get datasets by size category.
        
        Args:
            size_category: Size category ('small', 'medium', 'large')
            
        Returns:
            List of dataset information for the size category
        """
        return get_datasets_by_size(size_category)

    def register_custom_dataset(self, 
                               name: str,
                               description: str,
                               source_type: str,
                               source_config: Dict[str, Any],
                               size_estimate: Optional[int] = None,
                               splits: Optional[List[str]] = None,
                               tags: Optional[List[str]] = None,
                               citation: Optional[str] = None,
                               license: Optional[str] = None):
        """Register a custom dataset configuration.
        
        Args:
            name: Unique name for the dataset
            description: Description of the dataset
            source_type: Type of data source
            source_config: Configuration for the data source
            size_estimate: Estimated number of samples
            splits: Available splits
            tags: Tags for categorization
            citation: Citation information
            license: License information
        """
        dataset_info = DatasetInfo(
            name=name,
            description=description,
            source_type=source_type,
            source_config=source_config,
            size_estimate=size_estimate,
            splits=splits,
            tags=tags or [],
            citation=citation,
            license=license
        )
        
        self._custom_datasets[name] = dataset_info
        logger.info(f"Registered custom dataset: {name}")

    def discover_datasets(self) -> Dict[str, Any]:
        """Discover available datasets by category and use case.
        
        Returns:
            Dictionary with dataset discovery information
        """
        all_datasets = self.list_available_datasets()
        
        discovery = {
            "total_datasets": len(all_datasets),
            "predefined": len(DEFAULT_HUGGINGFACE_DATASETS) + len(BENCHMARK_DATASETS),
            "custom": len(self._custom_datasets),
            "categories": {},
            "use_cases": {},
            "size_distribution": {
                "small": len(self.get_datasets_by_size("small")),
                "medium": len(self.get_datasets_by_size("medium")),
                "large": len(self.get_datasets_by_size("large"))
            },
            "total_estimated_samples": estimate_total_size(list(all_datasets.keys()))
        }
        
        # Add category breakdown
        for category, info in DATASET_CATEGORIES.items():
            discovery["categories"][category] = {
                "description": info["description"],
                "datasets": info["datasets"],
                "count": len(info["datasets"])
            }
        
        # Add use case breakdown
        for use_case, info in USAGE_RECOMMENDATIONS.items():
            discovery["use_cases"][use_case] = {
                "description": info["description"],
                "recommended_datasets": info["recommended_datasets"],
                "considerations": info["considerations"],
                "count": len(info["recommended_datasets"])
            }
        
        return discovery

    def estimate_download_size(self, dataset_names: List[str]) -> Dict[str, Any]:
        """Estimate download size and time for datasets.
        
        Args:
            dataset_names: List of dataset names
            
        Returns:
            Dictionary with size estimates
        """
        total_samples = estimate_total_size(dataset_names)
        
        # Rough estimates (these would be more accurate with actual dataset metadata)
        avg_text_length = 500  # characters
        bytes_per_char = 1
        estimated_bytes = total_samples * avg_text_length * bytes_per_char
        estimated_mb = estimated_bytes / (1024 * 1024)
        
        return {
            "datasets": dataset_names,
            "estimated_samples": total_samples,
            "estimated_size_mb": estimated_mb,
            "estimated_size_gb": estimated_mb / 1024,
            "note": "Estimates are approximate and may vary significantly"
        }

    def list_categories(self) -> Dict[str, str]:
        """List available dataset categories.
        
        Returns:
            Dictionary mapping category names to descriptions
        """
        return {cat: info["description"] for cat, info in DATASET_CATEGORIES.items()}

    def list_use_cases(self) -> Dict[str, str]:
        """List available use cases.
        
        Returns:
            Dictionary mapping use case names to descriptions
        """
        return {case: info["description"] for case, info in USAGE_RECOMMENDATIONS.items()}

    def validate_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Validate a dataset configuration.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Validation results
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            return {"valid": False, "error": f"Dataset '{dataset_name}' not found"}
        
        validation = {
            "valid": True,
            "dataset_name": dataset_name,
            "source_type": dataset_info.source_type,
            "estimated_size": dataset_info.size_estimate,
            "available_splits": dataset_info.splits,
            "tags": dataset_info.tags
        }
        
        # Additional validation based on source type
        if dataset_info.source_type == "huggingface":
            required_keys = ["dataset_name", "prompt_column", "output_column", "label_column"]
            missing_keys = [key for key in required_keys if key not in dataset_info.source_config]
            if missing_keys:
                validation["valid"] = False
                validation["missing_config"] = missing_keys
        
        return validation

    def preview_dataset(self, dataset_name: str, num_samples: int = 5) -> Dict[str, Any]:
        """Get a preview of a dataset without loading it fully.
        
        Args:
            dataset_name: Name of the dataset
            num_samples: Number of samples to preview
            
        Returns:
            Preview information
        """
        try:
            # Load a small subset
            dataset = self.load_dataset(dataset_name, max_samples=num_samples)
            
            preview = {
                "dataset_name": dataset_name,
                "num_samples_shown": min(num_samples, len(dataset)),
                "total_estimated": self.get_dataset_info(dataset_name).size_estimate,
                "labels": dataset.get_labels(),
                "authors": dataset.get_authors(),
                "samples": []
            }
            
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                preview["samples"].append({
                    "prompt": sample.prompt[:100] + "..." if len(sample.prompt) > 100 else sample.prompt,
                    "output": sample.output[:200] + "..." if len(sample.output) > 200 else sample.output,
                    "author_id": sample.author_id,
                    "label": sample.label
                })
            
            return preview
            
        except Exception as e:
            return {
                "dataset_name": dataset_name,
                "error": str(e),
                "available": False
            } 