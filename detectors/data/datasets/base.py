"""Base dataset module for the framework.

This module provides the base classes and interfaces for dataset handling
in the framework, including the TextSample dataclass and BaseDataset abstract class.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, Union, Sequence
import pandas as pd
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)

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

class BaseDataset(ABC):
    """Base class for all datasets in the framework.
    
    This abstract class defines the interface that all dataset classes must implement.
    It provides common functionality for dataset operations and enforces a consistent
    interface across different dataset implementations.
    
    Attributes:
        data_path: Optional path to the dataset file or directory.
        samples: List of TextSample objects in the dataset.
    """
    
    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize the base dataset.
        
        Args:
            data_path: Optional path to the dataset file or directory.
        """
        self.data_path = Path(data_path) if data_path else None
        self.samples: List[TextSample] = []
        
    @abstractmethod
    def load(self) -> None:
        """Load the dataset from the source.
        
        This method must be implemented by subclasses to load data from their
        specific source (file, API, etc.).
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass
    
    def save(self, output_path: Optional[str] = None) -> None:
        """Save the dataset to disk.
        
        Args:
            output_path: Optional path to save the dataset to. If not provided,
                        uses the data_path attribute.
                        
        Raises:
            ValueError: If no output path is provided and data_path is None.
        """
        if output_path is None and self.data_path is None:
            raise ValueError("No output path provided and data_path is None")
            
        output_path = Path(output_path) if output_path else self.data_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [{
            'prompt': sample.prompt,
            'output': sample.output,
            'author_id': sample.author_id,
            'label': sample.label,
            'metadata': sample.metadata
        } for sample in self.samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(self.samples)} samples to {output_path}")
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame.
        
        Returns:
            A DataFrame containing all samples with their attributes and metadata.
        """
        return pd.DataFrame([{
            'prompt': sample.prompt,
            'output': sample.output,
            'author_id': sample.author_id,
            'label': sample.label,
            **(sample.metadata or {})
        } for sample in self.samples])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            The number of samples.
        """
        return len(self.samples)
    
    def _create_subset(self, samples: List[TextSample]) -> 'BaseDataset':
        """Create a new dataset instance with the given samples.
        
        This method should be overridden by subclasses to properly initialize
        the new dataset with required arguments.
        
        Args:
            samples: List of samples to include in the new dataset.
            
        Returns:
            A new dataset instance containing the given samples.
        """
        new_dataset = copy.copy(self)
        new_dataset.samples = samples
        return new_dataset
    
    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> Union[TextSample, 'BaseDataset']:
        """Get sample(s) by index, slice, or sequence of indices.
        
        Args:
            idx: Integer index, slice, or sequence of indices to get sample(s).
            
        Returns:
            A single TextSample, or a new BaseDataset containing the selected samples.
            
        Raises:
            IndexError: If any index is out of range.
            TypeError: If idx is not an integer, slice, or sequence of integers.
        """
        if isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, slice):
            # Create a new dataset with the sliced samples
            return self._create_subset(self.samples[idx])
        elif isinstance(idx, (list, tuple)) and all(isinstance(i, int) for i in idx):
            # Create a new dataset with the selected samples
            return self._create_subset([self.samples[i] for i in idx])
        else:
            raise TypeError("Index must be an integer, slice, or sequence of integers")
    
    def __iter__(self) -> Iterator[TextSample]:
        """Return an iterator over the samples.
        
        Returns:
            An iterator that yields TextSample objects.
        """
        return iter(self.samples)
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, random_state: int = 42) -> Dict[str, 'BaseDataset']:
        """Split the dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion of data to use for training.
            val_ratio: Proportion of data to use for validation.
            test_ratio: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Dictionary containing train, validation, and test datasets.
            
        Raises:
            ValueError: If ratios are invalid or don't sum to 1.
        """
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Split ratios must be between 0 and 1")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1")
            
        df = self.to_pandas()
        train_df = df.sample(frac=train_ratio, random_state=random_state)
        remaining = df.drop(train_df.index)
        val_df = remaining.sample(frac=val_ratio/(val_ratio + test_ratio), random_state=random_state)
        test_df = remaining.drop(val_df.index)
        
        splits = {}
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            dataset = self._create_subset([
                TextSample(
                    prompt=row['prompt'],
                    output=row['output'],
                    author_id=row['author_id'],
                    label=row['label'],
                    metadata={k: v for k, v in row.items() if k not in ['prompt', 'output', 'author_id', 'label']}
                )
                for _, row in split_df.iterrows()
            ])
            splits[name] = dataset
            
        return splits
        
    def get_samples_by_label(self, label: str) -> 'BaseDataset':
        """Get all samples with a specific label.
        
        Args:
            label: The label to filter samples by.
            
        Returns:
            New BaseDataset containing samples with the specified label.
        """
        return self._create_subset([sample for sample in self.samples if sample.label == label])
        
    def get_samples_by_author(self, author_id: str) -> 'BaseDataset':
        """Get all samples from a specific author.
        
        Args:
            author_id: The author ID to filter samples by.
            
        Returns:
            New BaseDataset containing samples from the specified author.
        """
        return self._create_subset([sample for sample in self.samples if sample.author_id == author_id])
        
    def add_sample(self, sample: TextSample) -> None:
        """Add a single sample to the dataset.
        
        Args:
            sample: The TextSample to add to the dataset.
        """
        self.samples.append(sample)
        
    def add_samples(self, samples: List[TextSample]) -> None:
        """Add multiple samples to the dataset.
        
        Args:
            samples: List of TextSample objects to add to the dataset.
        """
        self.samples.extend(samples)
        
    def clear(self) -> None:
        """Remove all samples from the dataset."""
        self.samples.clear() 