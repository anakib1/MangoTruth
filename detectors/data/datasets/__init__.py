"""Dataset module for the detectors framework.

This module provides a flexible dataset architecture with:
- Common interface for all datasets
- Pluggable storage backends (memory, pandas, etc.)
- Various data loaders (HuggingFace, JSON, folders, etc.)
- Factory functions for easy dataset creation
"""

# Core classes
from detectors.data.datasets.dataset import Dataset
from detectors.data.datasets.interfaces import TextSample

# Registry and default datasets
from detectors.data.datasets.registry import DatasetRegistry
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
    get_datasets_by_size
)

# Factory functions
from detectors.data.datasets.factory import (
    create_dataset,
    load_from_huggingface,
    convert_backend,
    merge_datasets
)

# Backends
from detectors.data.datasets.backends import (
    StorageBackend,
    InMemoryBackend,
    PandasBackend
)

# Loaders
from detectors.data.datasets.loaders import (
    DatasetLoader,
    HuggingFaceLoader
)

__all__ = [
    # Core
    'Dataset',
    'TextSample',
    
    # Registry and discovery
    'DatasetRegistry',
    'DatasetInfo',
    'DEFAULT_HUGGINGFACE_DATASETS',
    'BENCHMARK_DATASETS',
    'DATASET_CATEGORIES',
    'USAGE_RECOMMENDATIONS',
    'get_dataset_info',
    'list_available_datasets',
    'get_datasets_by_category',
    'get_datasets_by_tag',
    'get_recommended_datasets',
    'get_datasets_by_size',
    
    # Factory functions
    'create_dataset',
    'load_from_huggingface',
    'convert_backend',
    'merge_datasets',
    
    # Backends
    'StorageBackend',
    'InMemoryBackend',
    'PandasBackend',
    
    # Loaders
    'DatasetLoader',
    'HuggingFaceLoader'
]
