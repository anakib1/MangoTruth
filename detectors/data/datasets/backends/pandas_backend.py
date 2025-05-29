"""Pandas DataFrame storage backend implementation.

This module provides a pandas DataFrame-based storage backend for datasets.
"""

from typing import List, Union, Sequence, Dict, Any, Optional
import numpy as np
import pandas as pd
import sys
from detectors.data.datasets.backends.backend_interface import StorageBackend
from detectors.data.datasets.interfaces.dataset_interface import TextSample

class PandasBackend(StorageBackend[TextSample]):
    """Pandas DataFrame storage backend for TextSample data.
    
    This backend stores TextSample data in a pandas DataFrame for efficient
    operations on tabular data. Metadata is stored as JSON strings.
    
    Attributes:
        df: DataFrame storing all samples.
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        """Initialize the backend with optional DataFrame.
        
        Args:
            df: Optional initial DataFrame.
        """
        if df is None:
            self.df = pd.DataFrame(columns=['prompt', 'output', 'author_id', 'label', 'metadata'])
        else:
            self.df = df.copy()
    
    def __len__(self) -> int:
        """Return the number of items stored."""
        return len(self.df)
    
    def get_item(self, idx: int) -> TextSample:
        """Get a single item by index."""
        if idx < 0:
            idx = len(self.df) + idx
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
        
        row = self.df.iloc[idx]
        return self._row_to_sample(row)
    
    def get_items(self, indices: Union[slice, Sequence[int]]) -> List[TextSample]:
        """Get multiple items by indices."""
        if isinstance(indices, slice):
            rows = self.df.iloc[indices]
        else:
            rows = self.df.iloc[list(indices)]
        
        return [self._row_to_sample(row) for _, row in rows.iterrows()]
    
    def add_item(self, item: TextSample) -> None:
        """Add a single item to the backend."""
        new_row = pd.DataFrame([self._sample_to_dict(item)])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
    
    def add_items(self, items: List[TextSample]) -> None:
        """Add multiple items to the backend."""
        new_rows = pd.DataFrame([self._sample_to_dict(item) for item in items])
        self.df = pd.concat([self.df, new_rows], ignore_index=True)
    
    def delete_item(self, idx: int) -> None:
        """Delete an item by index."""
        self.df = self.df.drop(idx).reset_index(drop=True)
    
    def clear(self) -> None:
        """Remove all items from the backend."""
        self.df = pd.DataFrame(columns=['prompt', 'output', 'author_id', 'label', 'metadata'])
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        return self.df.memory_usage(deep=True).sum()
    
    def to_list(self) -> List[TextSample]:
        """Convert all stored items to a list."""
        return [self._row_to_sample(row) for _, row in self.df.iterrows()]
    
    def save(self, path: str) -> None:
        """Save the backend to disk using parquet format."""
        # Convert metadata to JSON strings for storage
        df_copy = self.df.copy()
        df_copy['metadata'] = df_copy['metadata'].apply(
            lambda x: pd.json_normalize(x).to_json() if isinstance(x, dict) else '{}'
        )
        df_copy.to_parquet(path, engine='pyarrow')
    
    def load(self, path: str) -> None:
        """Load the backend from disk using parquet format."""
        self.df = pd.read_parquet(path, engine='pyarrow')
        # Convert JSON strings back to dictionaries
        self.df['metadata'] = self.df['metadata'].apply(
            lambda x: pd.read_json(x).to_dict('records')[0] if x != '{}' else {}
        )
    
    def copy(self) -> 'PandasBackend':
        """Create a copy of the backend."""
        return PandasBackend(self.df.copy())
    
    def get_indices(self) -> np.ndarray:
        """Get array of valid indices."""
        return np.arange(len(self.df))
    
    def _sample_to_dict(self, sample: TextSample) -> Dict[str, Any]:
        """Convert TextSample to dictionary for DataFrame storage."""
        return {
            'prompt': sample.prompt,
            'output': sample.output,
            'author_id': sample.author_id,
            'label': sample.label,
            'metadata': sample.metadata or {}
        }
    
    def _row_to_sample(self, row: pd.Series) -> TextSample:
        """Convert DataFrame row to TextSample."""
        return TextSample(
            prompt=row['prompt'],
            output=row['output'],
            author_id=row['author_id'],
            label=row['label'],
            metadata=row.get('metadata', {})
        )
    
    # Additional pandas-specific methods
    def query(self, expr: str) -> 'PandasBackend':
        """Query the DataFrame using pandas query syntax.
        
        Args:
            expr: Query expression.
            
        Returns:
            New backend with filtered data.
        """
        return PandasBackend(self.df.query(expr))
    
    def groupby(self, by: Union[str, List[str]]) -> pd.core.groupby.DataFrameGroupBy:
        """Group the DataFrame by specified columns.
        
        Args:
            by: Column(s) to group by.
            
        Returns:
            GroupBy object.
        """
        return self.df.groupby(by)
    
    def get_label_counts(self) -> pd.Series:
        """Get counts of each label."""
        return self.df['label'].value_counts()
    
    def get_author_counts(self) -> pd.Series:
        """Get counts of each author."""
        return self.df['author_id'].value_counts() 