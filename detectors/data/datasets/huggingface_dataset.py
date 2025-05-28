from typing import Optional, Dict, Any, List, Union
import logging
from datasets import load_dataset, Dataset, DatasetDict, Features
from detectors.data.datasets.base import BaseDataset, TextSample

logger = logging.getLogger(__name__)

class HuggingFaceDataset(BaseDataset):
    """
    Dataset that loads samples from HuggingFace datasets.
    
    The dataset should have columns matching our TextSample structure:
    - prompt: input text
    - output: generated text
    - author_id: identifier for the author/model
    - label: classification label (e.g., "human", "gpt-4", "claude")
    
    Supports:
    - Multiple splits loading
    - Dataset configurations
    - Custom column mapping
    - Integer to string label mapping
    """
    
    def __init__(self, dataset_name: str, 
                 split: Optional[Union[str, List[str]]] = None,
                 config: Optional[str] = None,
                 prompt_column: str = "prompt", 
                 output_column: str = "output",
                 author_column: str = "author_id", 
                 label_column: str = "label",
                 label_mapping: Optional[Dict[int, str]] = None,
                 **kwargs):
        """
        Initialize HuggingFace dataset loader.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Single split name or list of split names to load
            config: Dataset configuration name (if dataset has multiple configs)
            prompt_column: Name of the column containing prompts
            output_column: Name of the column containing outputs
            author_column: Name of the column containing author IDs
            label_column: Name of the column containing labels
            label_mapping: Dictionary mapping integer labels to string labels
            **kwargs: Additional arguments to pass to load_dataset
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.kwargs = kwargs
        self.column_mapping = {
            'prompt': prompt_column,
            'output': output_column,
            'author_id': author_column,
            'label': label_column
        }
        self.label_mapping = label_mapping
        
    def load(self) -> None:
        """Load samples from HuggingFace dataset."""
        try:
            # Load the dataset
            dataset = load_dataset(
                self.dataset_name,
                self.config,
                split=self.split,
                **self.kwargs
            )
            
            # Get label mapping from dataset features if not provided
            if self.label_mapping is None and isinstance(dataset, (Dataset, DatasetDict)):
                self.label_mapping = self._extract_label_mapping(dataset)
            
            # Process the loaded dataset
            if isinstance(dataset, DatasetDict):
                self._process_dataset_dict(dataset)
            else:
                self.samples = self._process_dataset(dataset)
                
            logger.info(f"Loaded {len(self.samples)} samples from {self.dataset_name}" + 
                       (f" (config: {self.config})" if self.config else ""))
            if self.label_mapping:
                logger.info(f"Using label mapping: {self.label_mapping}")
        except Exception as e:
            raise ValueError(f"Failed to load dataset {self.dataset_name}" + 
                           (f" with config {self.config}" if self.config else "") + 
                           f": {e}")
            
    def _extract_label_mapping(self, dataset: Union[Dataset, DatasetDict]) -> Optional[Dict[int, str]]:
        """Extract label mapping from dataset features if available."""
        if isinstance(dataset, DatasetDict):
            # Use the first split to get features
            dataset = next(iter(dataset.values()))
            
        features = dataset.features
        label_feature = features.get(self.column_mapping['label'])
        
        if label_feature is None:
            return None
            
        # Check if the label feature has a mapping
        if hasattr(label_feature, 'names'):
            return {i: name for i, name in enumerate(label_feature.names)}
        elif hasattr(label_feature, 'int2str'):
            return label_feature.int2str
        elif hasattr(label_feature, 'mapping'):
            return label_feature.mapping
            
        return None
            
    def _process_dataset_dict(self, dataset_dict: DatasetDict) -> None:
        """Process a dictionary of datasets (multiple splits)."""
        self.samples = []
        for split_name, dataset in dataset_dict.items():
            split_samples = self._process_dataset(dataset)
            # Add split and config information to metadata
            for sample in split_samples:
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata['split'] = split_name
                if self.config:
                    sample.metadata['config'] = self.config
            self.samples.extend(split_samples)
            
    def _process_dataset(self, dataset: Dataset) -> List[TextSample]:
        """Convert HuggingFace dataset to our format."""
        samples = []
        
        for item in dataset:
            try:
                # Get the label and convert it if necessary
                label = item[self.column_mapping['label']]
                if isinstance(label, int) and self.label_mapping:
                    label = self.label_mapping[label]
                
                sample = TextSample(
                    prompt=item[self.column_mapping['prompt']],
                    output=item[self.column_mapping['output']],
                    author_id=str(item[self.column_mapping['author_id']]),
                    label=str(label),
                    metadata=self._extract_metadata(item)
                )
                samples.append(sample)
            except KeyError as e:
                logger.warning(f"Missing required column {e} in dataset item")
            except Exception as e:
                logger.warning(f"Failed to process dataset item: {e}")
                
        return samples
                
    def _extract_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional metadata from dataset item."""
        metadata = {}
        for key, value in item.items():
            if key not in self.column_mapping.values():
                metadata[key] = value
        return metadata
        
    @classmethod
    def from_huggingface_dataset(cls, dataset: Union[Dataset, DatasetDict], 
                               prompt_column: str = "prompt",
                               output_column: str = "output",
                               author_column: str = "author_id",
                               label_column: str = "label",
                               config: Optional[str] = None,
                               label_mapping: Optional[Dict[int, str]] = None) -> 'HuggingFaceDataset':
        """Create a dataset from an existing HuggingFace dataset object."""
        instance = cls("", prompt_column=prompt_column, output_column=output_column,
                      author_column=author_column, label_column=label_column,
                      config=config, label_mapping=label_mapping)
        
        if isinstance(dataset, DatasetDict):
            instance._process_dataset_dict(dataset)
        else:
            instance.samples = instance._process_dataset(dataset)
            
        return instance
        
    def get_samples_by_split(self, split: str) -> List[TextSample]:
        """Get all samples from a specific split."""
        return [sample for sample in self.samples 
                if sample.metadata and sample.metadata.get('split') == split]
        
    def get_samples_by_config(self, config: str) -> List[TextSample]:
        """Get all samples from a specific configuration."""
        return [sample for sample in self.samples 
                if sample.metadata and sample.metadata.get('config') == config]
        
    def get_available_splits(self) -> List[str]:
        """Get list of available splits in the dataset."""
        splits = set()
        for sample in self.samples:
            if sample.metadata and 'split' in sample.metadata:
                splits.add(sample.metadata['split'])
        return sorted(list(splits))
        
    def get_available_configs(self) -> List[str]:
        """Get list of available configurations in the dataset."""
        configs = set()
        for sample in self.samples:
            if sample.metadata and 'config' in sample.metadata:
                configs.add(sample.metadata['config'])
        return sorted(list(configs))
        
    def get_label_mapping(self) -> Optional[Dict[int, str]]:
        """Get the current label mapping."""
        return self.label_mapping 