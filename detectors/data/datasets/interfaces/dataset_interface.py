"""Dataset interface module.

This module defines the abstract interface that all dataset implementations must follow,
regardless of their storage backend or loading mechanism.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator, Union, Sequence, TypeVar, Generic
from dataclasses import dataclass, field

@dataclass
class TextSample:
    """Represents a single text sample in the dataset.
    
    This dataclass encapsulates all the information about a single text sample,
    including its prompt, output, author information, and metadata.
    
    Attributes:
        prompt: The input text/prompt for the sample.
        output: The generated/output text for the sample.
        author_id: Identifier for the author/model that generated the output.
        label: Classification label (e.g., "human", "gpt-4", "claude").
        metadata: Optional dictionary containing additional metadata about the sample.
    """
    prompt: str
    output: str
    author_id: str
    label: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

T = TypeVar('T')

class DatasetInterface(ABC, Generic[T]):
    """Abstract interface for all dataset implementations.
    
    This interface defines the common methods that all datasets must implement,
    regardless of their storage backend or loading mechanism.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> Union[T, 'DatasetInterface[T]']:
        """Get sample(s) by index, slice, or sequence of indices.
        
        Args:
            idx: Integer index, slice, or sequence of indices.
            
        Returns:
            Single sample or new dataset view with selected samples.
        """
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the samples."""
        pass
    
    @abstractmethod
    def filter(self, predicate: Any) -> 'DatasetInterface[T]':
        """Filter samples based on a predicate.
        
        Args:
            predicate: Function that returns True for samples to keep.
            
        Returns:
            New dataset view containing filtered samples.
        """
        pass
    
    @abstractmethod
    def map(self, function: Any) -> 'DatasetInterface[T]':
        """Apply a function to all samples.
        
        Args:
            function: Function to apply to each sample.
            
        Returns:
            New dataset view with transformed samples.
        """
        pass
    
    @abstractmethod
    def batch(self, batch_size: int) -> Iterator[List[T]]:
        """Return an iterator that yields batches of samples.
        
        Args:
            batch_size: Number of samples per batch.
            
        Yields:
            Lists of samples.
        """
        pass
    
    @abstractmethod
    def shuffle(self, seed: Optional[int] = None) -> 'DatasetInterface[T]':
        """Return a shuffled view of the dataset.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            New dataset view with shuffled samples.
        """
        pass
    
    @abstractmethod
    def split(self, ratios: List[float], seed: Optional[int] = None) -> List['DatasetInterface[T]']:
        """Split the dataset into multiple parts.
        
        Args:
            ratios: List of ratios for each split (must sum to 1).
            seed: Random seed for reproducibility.
            
        Returns:
            List of dataset views for each split.
        """
        pass

class TextDatasetInterface(DatasetInterface[TextSample]):
    """Interface specifically for text datasets."""
    
    @abstractmethod
    def get_by_label(self, label: str) -> 'TextDatasetInterface':
        """Get all samples with a specific label."""
        pass
    
    @abstractmethod
    def get_by_author(self, author_id: str) -> 'TextDatasetInterface':
        """Get all samples from a specific author."""
        pass
    
    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get a list of unique labels in the dataset."""
        pass
    
    @abstractmethod
    def get_authors(self) -> List[str]:
        """Get a list of unique authors in the dataset."""
        pass 