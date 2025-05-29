"""Main dataset implementation.

This module provides the main Dataset class that combines storage backends
and data loaders to provide a unified interface for dataset operations.
"""

from typing import List, Optional, Dict, Any, Iterator, Union, Sequence, Callable
import numpy as np
import random
from detectors.data.datasets.interfaces.dataset_interface import TextDatasetInterface, TextSample
from detectors.data.datasets.backends.backend_interface import StorageBackend
from detectors.data.datasets.backends.memory_backend import InMemoryBackend
from detectors.data.datasets.loaders.loader_interface import DatasetLoader

class Dataset(TextDatasetInterface):
    """Main dataset implementation with pluggable backends and loaders.
    
    This class provides a unified interface for working with datasets,
    supporting different storage backends and data loaders.
    
    Attributes:
        backend: Storage backend for the dataset.
        metadata: Metadata about the dataset.
    """
    
    def __init__(self, 
                 backend: Optional[StorageBackend[TextSample]] = None,
                 loader: Optional[DatasetLoader[TextSample]] = None,
                 load_immediately: bool = True) -> None:
        """Initialize the dataset.
        
        Args:
            backend: Storage backend to use (defaults to InMemoryBackend).
            loader: Optional loader to load data from.
            load_immediately: Whether to load data immediately if loader is provided.
        """
        if backend is not None:
            self.backend = backend
        else:
            self.backend = InMemoryBackend[TextSample]()
        self.metadata: Dict[str, Any] = {}
        
        if loader and load_immediately:
            loader.load(self.backend)
            self.metadata.update(loader.get_metadata())
    
    @classmethod
    def from_samples(cls, samples: List[TextSample]) -> 'Dataset':
        """Create a dataset from a list of samples.
        
        Args:
            samples: List of TextSample objects.
            
        Returns:
            New Dataset instance.
        """
        backend = InMemoryBackend[TextSample]()
        backend.add_items(samples)
        return cls(backend=backend)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.backend)
    
    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> Union[TextSample, 'Dataset']:
        """Get sample(s) by index, slice, or sequence of indices."""
        if isinstance(idx, int):
            return self.backend.get_item(idx)
        else:
            # Create a new dataset view with selected samples
            new_backend = InMemoryBackend[TextSample]()
            new_backend.add_items(self.backend.get_items(idx))
            new_dataset = Dataset(backend=new_backend)
            new_dataset.metadata = self.metadata.copy()
            return new_dataset
    
    def __iter__(self) -> Iterator[TextSample]:
        """Return an iterator over the samples."""
        for i in range(len(self)):
            yield self.backend.get_item(i)
    
    def filter(self, predicate: Callable[[TextSample], bool]) -> 'Dataset':
        """Filter samples based on a predicate."""
        filtered_samples = [
            sample for sample in self 
            if predicate(sample)
        ]
        return Dataset.from_samples(filtered_samples)
    
    def map(self, function: Callable[[TextSample], TextSample]) -> 'Dataset':
        """Apply a function to all samples."""
        mapped_samples = [function(sample) for sample in self]
        return Dataset.from_samples(mapped_samples)
    
    def batch(self, batch_size: int) -> Iterator[List[TextSample]]:
        """Return an iterator that yields batches of samples."""
        batch = []
        for sample in self:
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    def shuffle(self, seed: Optional[int] = None) -> 'Dataset':
        """Return a shuffled view of the dataset."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        
        shuffled_samples = self.backend.get_items(indices.tolist())
        return Dataset.from_samples(shuffled_samples)
    
    def split(self, ratios: List[float], seed: Optional[int] = None) -> List['Dataset']:
        """Split the dataset into multiple parts."""
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Shuffle indices
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        
        # Calculate split points
        split_points = []
        cumsum = 0
        for ratio in ratios[:-1]:
            cumsum += ratio
            split_points.append(int(cumsum * len(self)))
        
        # Create splits
        splits = []
        start = 0
        for end in split_points:
            split_indices = indices[start:end].tolist()
            split_samples = self.backend.get_items(split_indices)
            split_dataset = Dataset.from_samples(split_samples)
            split_dataset.metadata = self.metadata.copy()
            splits.append(split_dataset)
            start = end
        
        # Add last split
        split_indices = indices[start:].tolist()
        split_samples = self.backend.get_items(split_indices)
        split_dataset = Dataset.from_samples(split_samples)
        split_dataset.metadata = self.metadata.copy()
        splits.append(split_dataset)
        
        return splits
    
    def get_by_label(self, label: str) -> 'Dataset':
        """Get all samples with a specific label."""
        return self.filter(lambda sample: sample.label == label)
    
    def get_by_author(self, author_id: str) -> 'Dataset':
        """Get all samples from a specific author."""
        return self.filter(lambda sample: sample.author_id == author_id)
    
    def get_labels(self) -> List[str]:
        """Get list of unique labels in the dataset."""
        labels = set()
        for sample in self:
            labels.add(sample.label)
        return sorted(list(labels))
    
    def get_authors(self) -> List[str]:
        """Get list of unique authors in the dataset."""
        authors = set()
        for sample in self:
            authors.add(sample.author_id)
        return sorted(list(authors))
    
    def add_sample(self, sample: TextSample) -> None:
        """Add a single sample to the dataset."""
        self.backend.add_item(sample)
    
    def add_samples(self, samples: List[TextSample]) -> None:
        """Add multiple samples to the dataset."""
        self.backend.add_items(samples)
    
    def save(self, path: str) -> None:
        """Save the dataset to disk."""
        self.backend.save(path)
    
    def load(self, path: str) -> None:
        """Load the dataset from disk."""
        self.backend.load(path)
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        return self.backend.get_memory_usage()
    
    def to_list(self) -> List[TextSample]:
        """Convert the dataset to a list of samples."""
        return self.backend.to_list()
    
    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        backend_type = type(self.backend).__name__
        return f"Dataset(backend={backend_type}, size={len(self)}, labels={self.get_labels()})" 