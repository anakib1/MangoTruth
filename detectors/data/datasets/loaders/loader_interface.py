"""Loader interface module.

This module defines the abstract interface for dataset loaders.
Loaders are responsible for loading data from various sources.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from detectors.data.datasets.backends.backend_interface import StorageBackend

T = TypeVar('T')

class DatasetLoader(ABC, Generic[T]):
    """Abstract interface for dataset loaders.
    
    Loaders handle loading data from various sources (files, APIs, databases, etc.)
    and converting it into the appropriate format for storage backends.
    """
    
    @abstractmethod
    def load(self, backend: StorageBackend[T], **kwargs) -> None:
        """Load data into the provided storage backend.
        
        Args:
            backend: Storage backend to load data into.
            **kwargs: Additional loader-specific arguments.
        """
        pass
    
    @abstractmethod
    def can_load(self, source: Any) -> bool:
        """Check if this loader can handle the given source.
        
        Args:
            source: Source to check (path, URL, etc.).
            
        Returns:
            True if this loader can handle the source.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded data.
        
        Returns:
            Dictionary containing loader metadata.
        """
        pass
    
    @abstractmethod
    def estimate_size(self) -> Optional[int]:
        """Estimate the number of items that will be loaded.
        
        Returns:
            Estimated number of items, or None if unknown.
        """
        pass

class StreamingLoader(DatasetLoader[T]):
    """Abstract interface for streaming dataset loaders.
    
    Streaming loaders load data incrementally, which is useful for
    large datasets that don't fit in memory.
    """
    
    @abstractmethod
    def stream(self, batch_size: int = 1000) -> Any:
        """Stream data in batches.
        
        Args:
            batch_size: Number of items per batch.
            
        Yields:
            Batches of items.
        """
        pass
    
    def load(self, backend: StorageBackend[T], batch_size: int = 1000, **kwargs) -> None:
        """Load data into backend using streaming.
        
        Args:
            backend: Storage backend to load data into.
            batch_size: Number of items per batch.
            **kwargs: Additional loader-specific arguments.
        """
        for batch in self.stream(batch_size):
            backend.add_items(batch) 