"""Dataset processing utilities.

This module centralizes dataset processing logic that was previously
duplicated across multiple training scripts.
"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

from detectors.data.datasets import Dataset, TextSample

logger = logging.getLogger(__name__)


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    @abstractmethod
    def process(self, dataset: Any, max_samples: Optional[int] = None, seed: Optional[int] = None) -> Dataset:
        """Process raw data into our Dataset format.
        
        Args:
            dataset: Raw dataset to process.
            max_samples: Maximum number of samples to process.
            seed: Random seed for reproducibility.
            
        Returns:
            Processed Dataset instance.
        """
        pass


class HuggingFaceDatasetProcessor(DatasetProcessor):
    """Processor for HuggingFace datasets with comprehensive configuration options."""
    
    def __init__(self,
                 text_column: str = "output",
                 label_column: str = "label", 
                 prompt_column: str = "prompt",
                 author_column: Optional[str] = None,
                 label_mapping: Optional[Dict[int, str]] = None,
                 filter_labels: Optional[List[int]] = None,
                 balance_classes: bool = True):
        """Initialize the processor with configuration options.
        
        Args:
            text_column: Name of the text column in the dataset.
            label_column: Name of the label column in the dataset.
            prompt_column: Name of the prompt column in the dataset.
            author_column: Name of the author column (optional).
            label_mapping: Mapping from numeric labels to string labels.
            filter_labels: List of labels to keep (None = keep all).
            balance_classes: Whether to balance classes in the dataset.
        """
        self.text_column = text_column
        self.label_column = label_column
        self.prompt_column = prompt_column
        self.author_column = author_column
        self.label_mapping = label_mapping or {0: "human", 3: "ai"}
        self.filter_labels = filter_labels
        self.balance_classes = balance_classes
    
    def _map_label(self, numeric_label: int) -> str:
        """Map numeric label to string label.
        
        Args:
            numeric_label: Numeric label value.
            
        Returns:
            String label.
        """
        return self.label_mapping.get(numeric_label, str(numeric_label))
    
    def _filter_dataset(self, dataset):
        """Filter dataset by specified labels.
        
        Args:
            dataset: HuggingFace dataset to filter.
            
        Returns:
            Filtered dataset.
        """
        if self.filter_labels is None:
            return dataset
        
        def label_filter(example):
            return example[self.label_column] in self.filter_labels
        
        return dataset.filter(label_filter)
    
    def _balance_dataset(self, dataset):
        """Balance classes in the dataset.
        
        Args:
            dataset: HuggingFace dataset to balance.
            
        Returns:
            Balanced dataset.
        """
        if not self.balance_classes:
            return dataset
        
        try:
            # Get label distribution
            df = dataset.to_pandas()
            label_counts = df[self.label_column].value_counts()
            min_count = label_counts.min()
            
            # Sample equal number from each class
            balanced_indices = (
                df.groupby(self.label_column)
                .apply(lambda x: x.sample(min_count, random_state=42))
                .index.get_level_values(1)
                .values
            )
            
            return dataset.select(balanced_indices)
        except Exception as e:
            logger.warning(f"Failed to balance dataset: {e}")
            return dataset
    
    def _sample_dataset(self, dataset, max_samples: Optional[int], seed: Optional[int]):
        """Sample dataset to maximum number of samples.
        
        Args:
            dataset: HuggingFace dataset to sample.
            max_samples: Maximum samples to keep.
            seed: Random seed for sampling.
            
        Returns:
            Sampled dataset.
        """
        if max_samples is None or len(dataset) <= max_samples:
            return dataset
        
        shuffled = dataset.shuffle(seed=seed)
        return shuffled.select(range(max_samples))
    
    def _convert_to_text_samples(self, dataset) -> Dataset:
        """Convert HuggingFace dataset to TextSample objects.
        
        Args:
            dataset: HuggingFace dataset to convert.
            
        Returns:
            Dataset with TextSample objects.
        """
        samples = []
        
        for item in dataset:
            try:
                # Extract fields with defaults
                text = item.get(self.text_column, "")
                numeric_label = item.get(self.label_column, 0)
                prompt = item.get(self.prompt_column, None)
                author_id = item.get(self.author_column, None) if self.author_column else None
                
                # Map label
                label = self._map_label(numeric_label)
                
                # Create sample
                sample = TextSample(
                    prompt=prompt,
                    output=text,
                    author_id=author_id,
                    label=label
                )
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
        
        return Dataset.from_samples(samples)
    
    def process(self, dataset, max_samples: Optional[int] = None, seed: Optional[int] = None) -> Dataset:
        """Process HuggingFace dataset through the full pipeline.
        
        Args:
            dataset: HuggingFace dataset to process.
            max_samples: Maximum number of samples to keep.
            seed: Random seed for reproducibility.
            
        Returns:
            Processed Dataset instance.
        """
        logger.info(f"Processing dataset with {len(dataset)} samples")
        
        # Step 1: Filter by labels
        filtered_dataset = self._filter_dataset(dataset)
        logger.info(f"After filtering: {len(filtered_dataset)} samples")
        
        # Step 2: Balance classes
        balanced_dataset = self._balance_dataset(filtered_dataset)
        logger.info(f"After balancing: {len(balanced_dataset)} samples")
        
        # Step 3: Sample to max size
        sampled_dataset = self._sample_dataset(balanced_dataset, max_samples, seed)
        logger.info(f"After sampling: {len(sampled_dataset)} samples")
        
        # Step 4: Convert to our format
        final_dataset = self._convert_to_text_samples(sampled_dataset)
        logger.info(f"Final dataset: {len(final_dataset)} samples")
        
        return final_dataset


class DatasetProcessorFactory:
    """Factory for creating dataset processors."""
    
    _processors = {
        'huggingface': HuggingFaceDatasetProcessor,
    }
    
    @classmethod
    def create_processor(cls, processor_type: str, **kwargs) -> DatasetProcessor:
        """Create a dataset processor.
        
        Args:
            processor_type: Type of processor to create.
            **kwargs: Arguments to pass to processor constructor.
            
        Returns:
            Configured processor instance.
            
        Raises:
            ValueError: If processor type is not supported.
        """
        if processor_type not in cls._processors:
            available = ', '.join(cls._processors.keys())
            raise ValueError(f"Unsupported processor type: {processor_type}. "
                           f"Available: {available}")
        
        processor_class = cls._processors[processor_type]
        return processor_class(**kwargs)
    
    @classmethod
    def get_available_processors(cls) -> List[str]:
        """Get list of available processor types.
        
        Returns:
            List of available processor type names.
        """
        return list(cls._processors.keys())
    
    @classmethod
    def register_processor(cls, name: str, processor_class: type):
        """Register a new processor type.
        
        Args:
            name: Name for the processor type.
            processor_class: Processor class to register.
        """
        cls._processors[name] = processor_class 