"""Storage backend interface.

This module defines the abstract interface for dataset storage backends.
Backends are responsible for how data is actually stored and accessed.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator, Union, Sequence, TypeVar, Generic
import numpy as np

T = TypeVar('T')

class StorageBackend(ABC, Generic[T]):
    """Abstract interface for dataset storage backends.
    
    Storage backends handle the actual storage and retrieval of data,
    allowing for different implementations optimized for different use cases.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items stored."""
        pass
    
    @abstractmethod
    def get_item(self, idx: int) -> T:
        """Get a single item by index.
        
        Args:
            idx: Index of the item to retrieve.
            
        Returns:
            The item at the specified index.
        """
        pass
    
    @abstractmethod
    def get_items(self, indices: Union[slice, Sequence[int]]) -> List[T]:
        """Get multiple items by indices.
        
        Args:
            indices: Slice or sequence of indices.
            
        Returns:
            List of items at the specified indices.
        """
        pass
    
    @abstractmethod
    def add_item(self, item: T) -> None:
        """Add a single item to the backend.
        
        Args:
            item: Item to add.
        """
        pass
    
    @abstractmethod
    def add_items(self, items: List[T]) -> None:
        """Add multiple items to the backend.
        
        Args:
            items: List of items to add.
        """
        pass
    
    @abstractmethod
    def delete_item(self, idx: int) -> None:
        """Delete an item by index.
        
        Args:
            idx: Index of item to delete.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the backend."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        pass
    
    @abstractmethod
    def to_list(self) -> List[T]:
        """Convert all stored items to a list."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the backend to disk.
        
        Args:
            path: Path to save the backend.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the backend from disk.
        
        Args:
            path: Path to load the backend from.
        """
        pass
    
    @abstractmethod
    def copy(self) -> 'StorageBackend[T]':
        """Create a copy of the backend."""
        pass
    
    @abstractmethod
    def get_indices(self) -> np.ndarray:
        """Get array of valid indices.
        
        Returns:
            NumPy array of valid indices.
        """
        pass 