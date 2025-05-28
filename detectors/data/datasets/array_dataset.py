"""ArrayDataset module for in-memory dataset handling.

This module provides a simple dataset class for handling in-memory data,
particularly useful for generated data or data that doesn't need to be
loaded from disk.
"""

from typing import List, Optional
from detectors.data.datasets.base import BaseDataset, TextSample

class ArrayDataset(BaseDataset):
    """A simple dataset class for in-memory data.
    
    This class is designed for handling datasets that are kept in memory,
    such as generated data or data that doesn't need to be loaded from disk.
    It inherits all sample management functionality from BaseDataset and
    only implements the required load/save methods as no-ops.
    
    Attributes:
        samples: List of TextSample objects in the dataset (inherited from BaseDataset).
    """
    
    def __init__(self, samples: Optional[List[TextSample]] = None) -> None:
        """Initialize the dataset with an optional list of samples.
        
        Args:
            samples: Optional list of TextSample objects to initialize the dataset with.
                    If None, an empty list is used.
        """
        super().__init__()
        self.samples = samples or []
        
    def load(self) -> None:
        """No-op since data is already in memory.
        
        This method is implemented to satisfy the BaseDataset interface
        but does nothing since the data is already in memory.
        """
        pass
        
    def save(self) -> None:
        """No-op since data is kept in memory.
        
        This method is implemented to satisfy the BaseDataset interface
        but does nothing since the data is kept in memory.
        """
        pass 