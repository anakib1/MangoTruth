"""HuggingFace dataset loader implementation.

This module provides functionality to load datasets from the HuggingFace Hub.
"""

from typing import Optional, Dict, Any, List, Union
import logging
from datasets import load_dataset, Dataset, DatasetDict
from detectors.data.datasets.loaders.loader_interface import StreamingLoader
from detectors.data.datasets.backends.backend_interface import StorageBackend
from detectors.data.datasets.interfaces.dataset_interface import TextSample

logger = logging.getLogger(__name__)

class HuggingFaceLoader(StreamingLoader[TextSample]):
    """Loader for HuggingFace datasets.
    
    This loader handles loading datasets from the HuggingFace Hub,
    with support for various configurations and splits.
    
    Attributes:
        dataset_name: Name of the dataset on HuggingFace.
        config: Dataset configuration name.
        split: Dataset split(s) to load.
        column_mapping: Mapping from standard columns to dataset columns.
        label_mapping: Optional mapping from integer labels to string labels.
    """
    
    def __init__(self,
                 dataset_name: str,
                 config: Optional[str] = None,
                 split: Optional[Union[str, List[str]]] = None,
                 prompt_column: str = 'prompt',
                 output_column: str = 'output',
                 author_column: str = 'author_id',
                 label_column: str = 'label',
                 label_mapping: Optional[Dict[int, str]] = None,
                 **kwargs) -> None:
        """Initialize the HuggingFace loader.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace.
            config: Dataset configuration name.
            split: Single split or list of splits to load.
            prompt_column: Name of the prompt column in the dataset.
            output_column: Name of the output column in the dataset.
            author_column: Name of the author column in the dataset.
            label_column: Name of the label column in the dataset.
            label_mapping: Optional mapping from integer labels to strings.
            **kwargs: Additional arguments for load_dataset.
        """
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.column_mapping = {
            'prompt': prompt_column,
            'output': output_column,
            'author_id': author_column,
            'label': label_column
        }
        self.label_mapping = label_mapping
        self.kwargs = kwargs
        self._dataset = None
        self._metadata = {}
    
    def can_load(self, source: Any) -> bool:
        """Check if this loader can handle the given source."""
        return isinstance(source, str) and source == self.dataset_name
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded data."""
        return self._metadata
    
    def estimate_size(self) -> Optional[int]:
        """Estimate the number of items that will be loaded."""
        if self._dataset is None:
            return None
        
        if isinstance(self._dataset, DatasetDict):
            return sum(len(ds) for ds in self._dataset.values())
        else:
            return len(self._dataset)
    
    def stream(self, batch_size: int = 1000) -> Any:
        """Stream data in batches."""
        if self._dataset is None:
            self._load_dataset()
        
        # Handle DatasetDict (multiple splits)
        if isinstance(self._dataset, dict):  # Handle both DatasetDict and regular dict
            for split_name, dataset in self._dataset.items():
                yield from self._stream_dataset(dataset, split_name, batch_size)
        else:
            yield from self._stream_dataset(self._dataset, self.split, batch_size)
    
    def _load_dataset(self) -> None:
        """Load the dataset from HuggingFace."""
        try:
            self._dataset = load_dataset(
                self.dataset_name,
                self.config,
                split=self.split,
                **self.kwargs
            )
            
            # Extract label mapping if not provided
            if self.label_mapping is None:
                self.label_mapping = self._extract_label_mapping(self._dataset)
            
            # Store metadata
            self._metadata = {
                'dataset_name': self.dataset_name,
                'config': self.config,
                'split': self.split,
                'label_mapping': self.label_mapping
            }
            
            logger.info(f"Loaded dataset {self.dataset_name}"
                       + (f" (config: {self.config})" if self.config else ""))
            
        except Exception as e:
            raise ValueError(f"Failed to load dataset {self.dataset_name}: {e}")
    
    def _extract_label_mapping(self, dataset: Union[Dataset, DatasetDict]) -> Optional[Dict[int, str]]:
        """Extract label mapping from dataset features."""
        try:
            if isinstance(dataset, DatasetDict):
                # Use the first available dataset from the dict
                dataset = next(iter(dataset.values()))
            
            features = dataset.features
            label_feature = features.get(self.column_mapping['label'])
            
            if label_feature is None:
                return None
            
            if hasattr(label_feature, 'names'):
                return {i: name for i, name in enumerate(label_feature.names)}
            elif hasattr(label_feature, 'int2str'):
                return label_feature.int2str
            
            return None
        except Exception as e:
            logger.warning(f"Failed to extract label mapping: {e}")
            return None
    
    def _stream_dataset(self, dataset: Dataset, split_name: str, batch_size: int):
        """Stream a single dataset in batches."""
        batch = []
        
        for item in dataset:
            try:
                sample = self._convert_item(item, split_name or "unknown")
                batch.append(sample)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                    
            except Exception as e:
                logger.warning(f"Failed to convert item {item}: {e}")
        
        # Yield remaining items
        if batch:
            yield batch
    
    def _convert_item(self, item: Dict[str, Any], split_name: str) -> TextSample:
        """Convert a dataset item to TextSample."""
        # Ensure item is a dictionary
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict, got {type(item)}: {item}")
        
        # Get and convert label
        label_key = self.column_mapping['label']
        label = item.get(label_key, "unknown")
        if isinstance(label, int) and self.label_mapping:
            label = self.label_mapping.get(label, str(label))
        
        # Extract metadata
        metadata = {
            k: v for k, v in item.items()
            if k not in self.column_mapping.values()
        }
        metadata['split'] = split_name
        if self.config:
            metadata['config'] = self.config
        
        return TextSample(
            prompt=str(item.get(self.column_mapping['prompt'], "")),
            output=str(item.get(self.column_mapping['output'], "")),
            author_id=str(item.get(self.column_mapping['author_id'], "unknown")),
            label=str(label),
            metadata=metadata
        )
    
    def load(self, backend: StorageBackend[TextSample]) -> None:
        """Load data into the backend."""
        if self._dataset is None:
            self._load_dataset()
        
        # Load all data into backend
        for batch in self.stream(batch_size=1000):
            backend.add_items(batch) 