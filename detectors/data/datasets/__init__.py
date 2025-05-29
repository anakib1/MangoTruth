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
