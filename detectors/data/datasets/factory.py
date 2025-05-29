"""Dataset factory functions.

This module provides convenience functions for creating datasets with
different backends and loaders.
"""

from typing import Optional, List, Union, Dict, Any
from detectors.data.datasets.dataset import Dataset
from detectors.data.datasets.backends.memory_backend import InMemoryBackend
from detectors.data.datasets.backends.pandas_backend import PandasBackend
from detectors.data.datasets.loaders.huggingface_loader import HuggingFaceLoader
from detectors.data.datasets.interfaces.dataset_interface import TextSample

def create_dataset(backend_type: str = "memory", **kwargs) -> Dataset:
    """Create a dataset with the specified backend type.
    
    Args:
        backend_type: Type of backend ("memory", "pandas").
        **kwargs: Additional arguments for the backend.
        
    Returns:
        New Dataset instance.
    """
    if backend_type == "memory":
        backend = InMemoryBackend[TextSample]()
    elif backend_type == "pandas":
        backend = PandasBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return Dataset(backend=backend)

def load_from_huggingface(dataset_name: str,
                         backend_type: str = "memory",
                         config: Optional[str] = None,
                         split: Optional[Union[str, List[str]]] = None,
                         prompt_column: str = 'prompt',
                         output_column: str = 'output',
                         author_column: str = 'author_id',
                         label_column: str = 'label',
                         label_mapping: Optional[Dict[int, str]] = None,
                         **kwargs) -> Dataset:
    """Load a dataset from HuggingFace with the specified backend.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace.
        backend_type: Type of backend to use ("memory", "pandas").
        config: Dataset configuration name.
        split: Single split or list of splits to load.
        prompt_column: Name of the prompt column.
        output_column: Name of the output column.
        author_column: Name of the author column.
        label_column: Name of the label column.
        label_mapping: Optional mapping from integer labels to strings.
        **kwargs: Additional arguments for load_dataset.
        
    Returns:
        Dataset loaded from HuggingFace.
        
    Example:
        ```python
        # Load a dataset from HuggingFace
        dataset = load_from_huggingface(
            "truthful_qa",
            backend_type="pandas",
            config="generation",
            split="validation"
        )
        
        # Load with custom column mapping
        dataset = load_from_huggingface(
            "my_dataset",
            prompt_column="question",
            output_column="answer",
            label_column="source"
        )
        ```
    """
    # Create backend
    if backend_type == "memory":
        backend = InMemoryBackend[TextSample]()
    elif backend_type == "pandas":
        backend = PandasBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    # Create loader
    loader = HuggingFaceLoader(
        dataset_name=dataset_name,
        config=config,
        split=split,
        prompt_column=prompt_column,
        output_column=output_column,
        author_column=author_column,
        label_column=label_column,
        label_mapping=label_mapping,
        **kwargs
    )
    
    # Create and return dataset
    return Dataset(backend=backend, loader=loader)

def convert_backend(dataset: Dataset, new_backend_type: str) -> Dataset:
    """Convert a dataset to use a different backend.
    
    Args:
        dataset: Dataset to convert.
        new_backend_type: Type of backend to convert to.
        
    Returns:
        New Dataset with the specified backend.
        
    Example:
        ```python
        # Convert in-memory dataset to pandas backend
        pandas_dataset = convert_backend(memory_dataset, "pandas")
        ```
    """
    # Get all samples
    samples = dataset.to_list()
    
    # Create new backend
    if new_backend_type == "memory":
        new_backend = InMemoryBackend[TextSample]()
    elif new_backend_type == "pandas":
        new_backend = PandasBackend()
    else:
        raise ValueError(f"Unknown backend type: {new_backend_type}")
    
    # Add samples to new backend
    new_backend.add_items(samples)
    
    # Create new dataset
    new_dataset = Dataset(backend=new_backend)
    new_dataset.metadata = dataset.metadata.copy()
    
    return new_dataset

def merge_datasets(datasets: List[Dataset], backend_type: Optional[str] = None) -> Dataset:
    """Merge multiple datasets into one.
    
    Args:
        datasets: List of datasets to merge.
        backend_type: Type of backend for the merged dataset (uses first dataset's type if None).
        
    Returns:
        Merged dataset.
        
    Example:
        ```python
        # Merge multiple datasets
        merged = merge_datasets([dataset1, dataset2, dataset3])
        ```
    """
    if not datasets:
        raise ValueError("No datasets provided to merge")
    
    # Determine backend type
    if backend_type is None:
        # Use the backend type of the first dataset
        first_backend = datasets[0].backend
        if isinstance(first_backend, PandasBackend):
            backend_type = "pandas"
        else:
            backend_type = "memory"
    
    # Create new dataset
    merged = create_dataset(backend_type)
    
    # Add all samples
    for dataset in datasets:
        merged.add_samples(dataset.to_list())
    
    # Merge metadata
    merged.metadata = {
        'merged_from': [d.metadata for d in datasets],
        'num_sources': len(datasets)
    }
    
    return merged 