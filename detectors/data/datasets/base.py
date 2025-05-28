from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class TextSample:
    """Represents a single text sample in the dataset."""
    prompt: str
    output: str
    author_id: str
    label: str
    metadata: Optional[Dict[str, Any]] = None

class BaseDataset(ABC):
    """Base class for all datasets in the framework."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None
        self.samples: List[TextSample] = []
        
    @abstractmethod
    def load(self) -> None:
        """Load the dataset from the source."""
        pass
    
    def save(self, output_path: str) -> None:
        """Save the dataset to disk."""
        output_path = Path(output_path)
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
        """Convert the dataset to a pandas DataFrame."""
        return pd.DataFrame([{
            'prompt': sample.prompt,
            'output': sample.output,
            'author_id': sample.author_id,
            'label': sample.label,
            **(sample.metadata or {})
        } for sample in self.samples])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> TextSample:
        return self.samples[idx]
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, random_state: int = 42) -> Dict[str, 'BaseDataset']:
        """Split the dataset into train, validation, and test sets."""
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
            dataset = self.__class__()
            dataset.samples = [
                TextSample(
                    prompt=row['prompt'],
                    output=row['output'],
                    author_id=row['author_id'],
                    label=row['label'],
                    metadata={k: v for k, v in row.items() if k not in ['prompt', 'output', 'author_id', 'label']}
                )
                for _, row in split_df.iterrows()
            ]
            splits[name] = dataset
            
        return splits 