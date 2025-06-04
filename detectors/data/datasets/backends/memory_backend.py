"""In-memory storage backend implementation.

This module provides a simple list-based storage backend for datasets.
"""

from typing import List, Union, Sequence, TypeVar
import numpy as np
import pickle
import sys
from copy import deepcopy
from detectors.data.datasets.backends.backend_interface import StorageBackend

T = TypeVar('T')

class InMemoryBackend(StorageBackend[T]):
    """Simple in-memory storage backend using Python lists.
    
    This backend stores all data in memory using a Python list.
    It's suitable for small to medium datasets that fit in memory.
    
    Attributes:
        data: List storing all items.
    """
    
    def __init__(self, data: List[T] = None) -> None:
        """Initialize the backend with optional initial data.
        
        Args:
            data: Optional initial data.
        """
        self.data: List[T] = data if data is not None else []
    
    def __len__(self) -> int:
        """Return the number of items stored."""
        return len(self.data)
    
    def get_item(self, idx: int) -> T:
        """Get a single item by index."""
        if idx < 0:
            idx = len(self.data) + idx
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]
    
    def get_items(self, indices: Union[slice, Sequence[int]]) -> List[T]:
        """Get multiple items by indices."""
        if isinstance(indices, slice):
            return self.data[indices]
        else:
            return [self.data[i] for i in indices]
    
    def add_item(self, item: T) -> None:
        """Add a single item to the backend."""
        self.data.append(item)
    
    def add_items(self, items: List[T]) -> None:
        """Add multiple items to the backend."""
        self.data.extend(items)
    
    def delete_item(self, idx: int) -> None:
        """Delete an item by index."""
        del self.data[idx]
    
    def clear(self) -> None:
        """Remove all items from the backend."""
        self.data.clear()
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        return sys.getsizeof(self.data) + sum(sys.getsizeof(item) for item in self.data)
    
    def to_list(self) -> List[T]:
        """Convert all stored items to a list."""
        return self.data.copy()
    
    def save(self, path: str) -> None:
        """Save the backend to disk using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
    
    def load(self, path: str) -> None:
        """Load the backend from the disk using pickle."""
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
    
    def copy(self) -> 'InMemoryBackend[T]':
        """Create a deep copy of the backend."""
        return InMemoryBackend(deepcopy(self.data))
    
    def get_indices(self) -> np.ndarray:
        """Get an array of valid indices."""
        return np.arange(len(self.data)) 