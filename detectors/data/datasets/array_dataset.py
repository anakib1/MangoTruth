from typing import List, Optional
from detectors.data.datasets.base import BaseDataset, TextSample

class ArrayDataset(BaseDataset):
    """
    A simple dataset class for in-memory data.
    This is useful for generated data or data that doesn't need to be loaded from disk.
    """
    
    def __init__(self, samples: Optional[List[TextSample]] = None):
        """
        Initialize the dataset with an optional list of samples.
        
        Args:
            samples: Optional list of TextSample objects to initialize the dataset with
        """
        super().__init__()
        self.samples = samples or []
        
    def load(self) -> None:
        """No-op since data is already in memory."""
        pass
        
    def save(self) -> None:
        """No-op since data is kept in memory."""
        pass 