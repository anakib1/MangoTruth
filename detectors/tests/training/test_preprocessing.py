"""Tests for dataset preprocessing components."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from detectors.training.preprocessing import (
    DatasetProcessorFactory,
    HuggingFaceDatasetProcessor,
    DatasetProcessor
)
from detectors.data.datasets import Dataset, TextSample


# Simple test class that mimics HuggingFace dataset behavior
class SimpleTestDataset:
    def __init__(self, data, length=None):
        self.data = data
        self._length = length or len(data)
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        return iter(self.data)
    
    def filter(self, func):
        filtered_data = [item for item in self.data if func(item)]
        return SimpleTestDataset(filtered_data)
    
    def shuffle(self, seed=None):
        return self  # Return self for simplicity
    
    def select(self, indices):
        selected_data = [self.data[i] for i in indices if i < len(self.data)]
        return SimpleTestDataset(selected_data)
    
    def to_pandas(self):
        # Mock pandas operations for balancing
        mock_df = Mock()
        mock_series = Mock()
        mock_series.value_counts.return_value = Mock()
        mock_series.value_counts.return_value.min.return_value = min(2, len(self.data))
        
        # Mock groupby operations
        mock_grouped = Mock()
        mock_applied = Mock()
        mock_index = Mock()
        mock_level_values = Mock()
        mock_level_values.values = list(range(len(self.data)))
        mock_index.get_level_values.return_value = mock_level_values
        mock_applied.index = mock_index
        mock_grouped.apply.return_value = mock_applied
        mock_df.groupby.return_value = mock_grouped
        
        mock_df.__getitem__ = Mock(return_value=mock_series)
        return mock_df


class TestHuggingFaceDatasetProcessor:
    """Tests for HuggingFaceDatasetProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization with default parameters."""
        processor = HuggingFaceDatasetProcessor()
        
        assert processor.text_column == "output"
        assert processor.label_column == "label"
        assert processor.prompt_column == "prompt"
        assert processor.author_column is None
        assert processor.label_mapping == {0: "human", 3: "ai"}
        assert processor.filter_labels is None
        assert processor.balance_classes is True
    
    def test_initialization_custom_parameters(self):
        """Test processor initialization with custom parameters."""
        custom_mapping = {0: "real", 1: "fake"}
        processor = HuggingFaceDatasetProcessor(
            text_column="text",
            label_column="class",
            prompt_column="question",
            author_column="user_id",
            label_mapping=custom_mapping,
            filter_labels=[0, 1],
            balance_classes=False
        )
        
        assert processor.text_column == "text"
        assert processor.label_column == "class"
        assert processor.prompt_column == "question"
        assert processor.author_column == "user_id"
        assert processor.label_mapping == custom_mapping
        assert processor.filter_labels == [0, 1]
        assert processor.balance_classes is False
    
    def test_map_labels(self):
        """Test label mapping functionality."""
        processor = HuggingFaceDatasetProcessor()
        
        # Test default mapping
        assert processor._map_label(0) == "human"
        assert processor._map_label(3) == "ai"
        
        # Test unmapped label (should convert to string)
        assert processor._map_label(999) == "999"
        
        # Test custom mapping
        custom_processor = HuggingFaceDatasetProcessor(
            label_mapping={1: "positive", 2: "negative"}
        )
        assert custom_processor._map_label(1) == "positive"
        assert custom_processor._map_label(2) == "negative"
        assert custom_processor._map_label(999) == "999"
    
    def test_filter_dataset(self):
        """Test dataset filtering by labels."""
        processor = HuggingFaceDatasetProcessor(filter_labels=[0])
        
        test_data = [
            {"output": "Text 1", "label": 0},
            {"output": "Text 2", "label": 1},
            {"output": "Text 3", "label": 0}
        ]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._filter_dataset(dataset)
        
        # Should only have items with label 0
        result_data = list(result)
        assert len(result_data) == 2
        assert all(item["label"] == 0 for item in result_data)
    
    def test_filter_dataset_no_filter(self):
        """Test dataset when no filtering is applied."""
        processor = HuggingFaceDatasetProcessor(filter_labels=None)
        
        test_data = [{"output": "Text 1", "label": 0}]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._filter_dataset(dataset)
        
        # Should return same data
        assert list(result) == test_data
    
    def test_convert_to_text_samples(self):
        """Test conversion to TextSample objects."""
        processor = HuggingFaceDatasetProcessor(author_column="user_id")
        
        test_data = [
            {"output": "Text 1", "label": 0, "prompt": "Prompt 1", "user_id": "user1"},
            {"output": "Text 2", "label": 3, "prompt": "Prompt 2", "user_id": "user2"}
        ]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._convert_to_text_samples(dataset)
        
        # Check that we got a Dataset with TextSamples
        assert isinstance(result, Dataset)
        samples = result.to_list()
        assert len(samples) == 2
        
        # Check first sample
        sample1 = samples[0]
        assert sample1.output == "Text 1"
        assert sample1.label == "human"  # Mapped from 0
        assert sample1.prompt == "Prompt 1"
        assert sample1.author_id == "user1"
        
        # Check second sample
        sample2 = samples[1]
        assert sample2.output == "Text 2"
        assert sample2.label == "ai"  # Mapped from 3
        assert sample2.prompt == "Prompt 2"
        assert sample2.author_id == "user2"
    
    def test_convert_to_text_samples_missing_columns(self):
        """Test conversion when some columns are missing."""
        processor = HuggingFaceDatasetProcessor(author_column=None)
        
        test_data = [
            {"output": "Text 1", "label": 0, "prompt": "Prompt 1"},
            {"output": "Text 2", "label": 3}  # Missing prompt
        ]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._convert_to_text_samples(dataset)
        
        samples = result.to_list()
        assert len(samples) == 2
        
        sample1 = samples[0]
        assert sample1.author_id is None
        
        sample2 = samples[1]
        assert sample2.prompt is None
        assert sample2.author_id is None
    
    def test_sample_dataset_with_limit(self):
        """Test dataset sampling when max_samples is specified."""
        processor = HuggingFaceDatasetProcessor()
        
        # Create dataset with more items than limit
        test_data = [{"output": f"Text {i}", "label": 0} for i in range(10)]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._sample_dataset(dataset, max_samples=5, seed=42)
        
        # Should be limited to 5 items
        assert len(result) == 5
    
    def test_sample_dataset_no_limit(self):
        """Test dataset sampling when max_samples is None."""
        processor = HuggingFaceDatasetProcessor()
        
        test_data = [{"output": "Text 1", "label": 0}]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._sample_dataset(dataset, max_samples=None, seed=42)
        
        # Should return original dataset
        assert result is dataset
    
    def test_sample_dataset_smaller_than_limit(self):
        """Test dataset sampling when dataset is smaller than max_samples."""
        processor = HuggingFaceDatasetProcessor()
        
        test_data = [{"output": "Text 1", "label": 0}]
        dataset = SimpleTestDataset(test_data)
        
        result = processor._sample_dataset(dataset, max_samples=1000, seed=42)
        
        # Should return original dataset
        assert result is dataset
    
    def test_process_full_workflow(self):
        """Test complete processing workflow."""
        processor = HuggingFaceDatasetProcessor(
            filter_labels=[0, 3],
            balance_classes=False  # Disable balancing for simpler test
        )
        
        test_data = [
            {"output": "Human text", "label": 0, "prompt": "Write something"},
            {"output": "AI text", "label": 3, "prompt": "Write something"},
            {"output": "Other text", "label": 1, "prompt": "Write something"}  # Will be filtered out
        ]
        dataset = SimpleTestDataset(test_data)
        
        result = processor.process(dataset, max_samples=2, seed=42)
        
        # Should have filtered and processed the data
        assert isinstance(result, Dataset)
        samples = result.to_list()
        assert len(samples) == 2  # Only the items with labels 0 and 3
        
        # Check labels were mapped correctly
        labels = [sample.label for sample in samples]
        assert "human" in labels
        assert "ai" in labels


class TestDatasetProcessorFactory:
    """Tests for DatasetProcessorFactory class."""
    
    def test_create_huggingface_processor(self):
        """Test creating HuggingFace processor."""
        processor = DatasetProcessorFactory.create_processor("huggingface")
        
        assert isinstance(processor, HuggingFaceDatasetProcessor)
        assert processor.text_column == "output"
        assert processor.label_column == "label"
    
    def test_create_huggingface_processor_with_kwargs(self):
        """Test creating HuggingFace processor with custom parameters."""
        processor = DatasetProcessorFactory.create_processor(
            "huggingface",
            text_column="text",
            label_mapping={1: "positive", 0: "negative"},
            balance_classes=False
        )
        
        assert isinstance(processor, HuggingFaceDatasetProcessor)
        assert processor.text_column == "text"
        assert processor.label_mapping == {1: "positive", 0: "negative"}
        assert processor.balance_classes is False
    
    def test_create_unsupported_processor(self):
        """Test error handling for unsupported processor types."""
        with pytest.raises(ValueError, match="Unsupported processor type"):
            DatasetProcessorFactory.create_processor("unsupported_type")
    
    def test_get_available_processors(self):
        """Test getting available processor types."""
        available = DatasetProcessorFactory.get_available_processors()
        
        assert isinstance(available, list)
        assert "huggingface" in available
    
    def test_register_processor(self):
        """Test registering a new processor type."""
        class CustomProcessor(DatasetProcessor):
            def process(self, dataset, max_samples=None, seed=None):
                return dataset
        
        # Register the processor
        DatasetProcessorFactory.register_processor("custom", CustomProcessor)
        
        # Test creation
        processor = DatasetProcessorFactory.create_processor("custom")
        assert isinstance(processor, CustomProcessor)
        
        # Test it appears in available processors
        available = DatasetProcessorFactory.get_available_processors()
        assert "custom" in available
        
        # Clean up
        if "custom" in DatasetProcessorFactory._processors:
            del DatasetProcessorFactory._processors["custom"]


class TestDatasetProcessorIntegration:
    """Integration tests for dataset processing."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end dataset processing."""
        test_data = [
            {"output": "AI generated text", "label": 3, "prompt": "Write about AI"},
            {"output": "Human written text", "label": 0, "prompt": "Write about humans"},
            {"output": "Another human text", "label": 0, "prompt": "Write more"},
            {"output": "Another AI text", "label": 3, "prompt": "Write more"}
        ]
        dataset = SimpleTestDataset(test_data)
        
        processor = DatasetProcessorFactory.create_processor(
            "huggingface",
            filter_labels=[0, 3],
            balance_classes=False  # Simplify for testing
        )
        
        result = processor.process(dataset, max_samples=3, seed=42)
        
        # Verify final result
        assert isinstance(result, Dataset)
        samples = result.to_list()
        assert len(samples) <= 3  # Should be limited by max_samples
        
        # Check sample properties
        for sample in samples:
            assert sample.label in ["human", "ai"]
            assert sample.output is not None
            assert sample.prompt is not None
    
    def test_processor_with_custom_config(self):
        """Test processor with custom configuration."""
        test_data = [
            {"text": "Sample text", "class": 1, "id": "1"},
        ]
        dataset = SimpleTestDataset(test_data)
        
        processor = DatasetProcessorFactory.create_processor(
            "huggingface",
            text_column="text",
            label_column="class",
            author_column="id",
            label_mapping={1: "positive"}
        )
        
        result = processor.process(dataset)
        
        # Check the sample was created correctly
        samples = result.to_list()
        assert len(samples) == 1
        assert samples[0].output == "Sample text"
        assert samples[0].label == "positive"
        assert samples[0].author_id == "1"
    
    def test_empty_dataset(self):
        """Test processing an empty dataset."""
        dataset = SimpleTestDataset([])
        processor = HuggingFaceDatasetProcessor()
        
        result = processor.process(dataset)
        
        assert isinstance(result, Dataset)
        assert len(result) == 0
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data samples."""
        test_data = [
            {"output": "Good sample", "label": 0},
            {"incomplete": "sample"},  # Missing required fields
        ]
        dataset = SimpleTestDataset(test_data)
        
        processor = HuggingFaceDatasetProcessor()
        
        # Should handle gracefully (implementation may skip malformed samples)
        result = processor.process(dataset)
        
        # At minimum, should not crash and should process the good sample
        assert isinstance(result, Dataset)
        assert len(result) >= 1 