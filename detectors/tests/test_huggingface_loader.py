"""Unit tests for HuggingFace loader.

This module contains tests for the HuggingFaceLoader class,
including data loading and conversion functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from detectors.data.datasets.loaders import HuggingFaceLoader
from detectors.data.datasets.backends import InMemoryBackend
from detectors.data.datasets.interfaces import TextSample

class TestHuggingFaceLoader(unittest.TestCase):
    """Test suite for HuggingFaceLoader."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_dataset_item = {
            'prompt': 'Test prompt',
            'output': 'Test output',
            'author_id': 'test_author',
            'label': 0,  # Integer label
            'extra_field': 'extra_value'
        }
        
        self.mock_dataset = Mock()
        self.mock_dataset.__iter__ = Mock(return_value=iter([self.mock_dataset_item]))
        self.mock_dataset.__len__ = Mock(return_value=1)
        self.mock_dataset.features = {
            'label': Mock(names=['human', 'ai'])  # Mock ClassLabel feature
        }
    
    def test_init(self):
        """Test loader initialization."""
        loader = HuggingFaceLoader(
            dataset_name="test_dataset",
            config="test_config",
            split="train",
            prompt_column="question",
            output_column="answer",
            author_column="source",
            label_column="category"
        )
        
        self.assertEqual(loader.dataset_name, "test_dataset")
        self.assertEqual(loader.config, "test_config")
        self.assertEqual(loader.split, "train")
        self.assertEqual(loader.column_mapping['prompt'], "question")
        self.assertEqual(loader.column_mapping['output'], "answer")
        self.assertEqual(loader.column_mapping['author_id'], "source")
        self.assertEqual(loader.column_mapping['label'], "category")
    
    def test_can_load(self):
        """Test can_load method."""
        loader = HuggingFaceLoader("test_dataset")
        
        self.assertTrue(loader.can_load("test_dataset"))
        self.assertFalse(loader.can_load("other_dataset"))
        self.assertFalse(loader.can_load(123))
    
    def test_estimate_size_no_dataset(self):
        """Test estimate_size when dataset not loaded."""
        loader = HuggingFaceLoader("test_dataset")
        self.assertIsNone(loader.estimate_size())
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        """Test loading dataset from HuggingFace."""
        mock_load_dataset.return_value = self.mock_dataset
        
        loader = HuggingFaceLoader("test_dataset", config="test_config", split="train")
        backend = InMemoryBackend()
        
        loader.load(backend)
        
        # Verify load_dataset was called with correct arguments
        mock_load_dataset.assert_called_once_with(
            "test_dataset",
            "test_config",
            split="train"
        )
        
        # Verify data was loaded into backend
        self.assertEqual(len(backend), 1)
        sample = backend.get_item(0)
        self.assertIsInstance(sample, TextSample)
        self.assertEqual(sample.prompt, "Test prompt")
        self.assertEqual(sample.output, "Test output")
        self.assertEqual(sample.author_id, "test_author")
        self.assertEqual(sample.label, "human")  # Should be converted from integer
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_load_dataset_error(self, mock_load_dataset):
        """Test error handling during dataset loading."""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        loader = HuggingFaceLoader("nonexistent_dataset")
        backend = InMemoryBackend()
        
        with self.assertRaises(ValueError) as context:
            loader.load(backend)
        
        self.assertIn("Failed to load dataset", str(context.exception))
        self.assertIn("nonexistent_dataset", str(context.exception))
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_extract_label_mapping(self, mock_load_dataset):
        """Test extracting label mapping from dataset features."""
        mock_load_dataset.return_value = self.mock_dataset
        
        loader = HuggingFaceLoader("test_dataset")
        loader._load_dataset()
        
        # Should extract label mapping from features
        expected_mapping = {0: 'human', 1: 'ai'}
        self.assertEqual(loader.label_mapping, expected_mapping)
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_custom_label_mapping(self, mock_load_dataset):
        """Test using custom label mapping."""
        mock_load_dataset.return_value = self.mock_dataset
        
        custom_mapping = {0: 'person', 1: 'machine'}
        loader = HuggingFaceLoader("test_dataset", label_mapping=custom_mapping)
        backend = InMemoryBackend()
        
        loader.load(backend)
        
        sample = backend.get_item(0)
        self.assertEqual(sample.label, "person")  # Should use custom mapping
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_convert_item(self, mock_load_dataset):
        """Test converting dataset item to TextSample."""
        mock_load_dataset.return_value = self.mock_dataset
        
        loader = HuggingFaceLoader("test_dataset")
        loader.label_mapping = {0: 'human', 1: 'ai'}
        
        sample = loader._convert_item(self.mock_dataset_item, "train")
        
        self.assertIsInstance(sample, TextSample)
        self.assertEqual(sample.prompt, "Test prompt")
        self.assertEqual(sample.output, "Test output")
        self.assertEqual(sample.author_id, "test_author")
        self.assertEqual(sample.label, "human")
        
        # Check metadata
        self.assertEqual(sample.metadata['split'], "train")
        self.assertEqual(sample.metadata['extra_field'], "extra_value")
        self.assertNotIn('prompt', sample.metadata)  # Standard fields excluded
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_stream_single_dataset(self, mock_load_dataset):
        """Test streaming from single dataset."""
        # Create multiple items for batching test
        items = [
            {'prompt': f'Q{i}', 'output': f'A{i}', 'author_id': f'author{i}', 'label': i % 2}
            for i in range(5)
        ]
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(items))
        mock_dataset.features = {'label': Mock(names=['human', 'ai'])}
        mock_load_dataset.return_value = mock_dataset
        
        loader = HuggingFaceLoader("test_dataset")
        
        # Test streaming with batch size 2
        batches = list(loader.stream(batch_size=2))
        
        self.assertEqual(len(batches), 3)  # 5 items -> 3 batches (2, 2, 1)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 2)
        self.assertEqual(len(batches[2]), 1)
        
        # Verify first batch content
        first_batch = batches[0]
        self.assertEqual(first_batch[0].prompt, "Q0")
        self.assertEqual(first_batch[1].prompt, "Q1")
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_stream_dataset_dict(self, mock_load_dataset):
        """Test streaming from DatasetDict with multiple splits."""
        from datasets import DatasetDict
        
        # Mock train and validation datasets
        train_items = [{'prompt': 'Train Q', 'output': 'Train A', 'author_id': 'author', 'label': 0}]
        val_items = [{'prompt': 'Val Q', 'output': 'Val A', 'author_id': 'author', 'label': 1}]
        
        train_dataset = Mock()
        train_dataset.__iter__ = Mock(return_value=iter(train_items))
        train_dataset.features = {'label': Mock(names=['human', 'ai'])}
        
        val_dataset = Mock()
        val_dataset.__iter__ = Mock(return_value=iter(val_items))
        val_dataset.features = {'label': Mock(names=['human', 'ai'])}
        
        dataset_dict = {'train': train_dataset, 'validation': val_dataset}
        mock_load_dataset.return_value = dataset_dict
        
        loader = HuggingFaceLoader("test_dataset")
        
        # Stream all data
        all_batches = list(loader.stream(batch_size=10))
        
        # Should get batches from both splits
        all_samples = []
        for batch in all_batches:
            all_samples.extend(batch)
        
        self.assertEqual(len(all_samples), 2)
        
        # Check that samples have correct split metadata
        splits = [sample.metadata['split'] for sample in all_samples]
        self.assertIn('train', splits)
        self.assertIn('validation', splits)
    
    def test_get_metadata(self):
        """Test getting loader metadata."""
        loader = HuggingFaceLoader("test_dataset", config="test_config", split="train")
        loader._metadata = {
            'dataset_name': 'test_dataset',
            'config': 'test_config',
            'split': 'train'
        }
        
        metadata = loader.get_metadata()
        self.assertEqual(metadata['dataset_name'], 'test_dataset')
        self.assertEqual(metadata['config'], 'test_config')
        self.assertEqual(metadata['split'], 'train')
    
    @patch('detectors.data.datasets.loaders.huggingface_loader.load_dataset')
    def test_column_mapping(self, mock_load_dataset):
        """Test custom column mapping."""
        custom_item = {
            'question': 'Test question',
            'answer': 'Test answer',
            'source': 'test_source',
            'category': 'test_category',
            'metadata_field': 'metadata_value'
        }
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([custom_item]))
        mock_dataset.features = {}
        mock_load_dataset.return_value = mock_dataset
        
        loader = HuggingFaceLoader(
            "test_dataset",
            prompt_column="question",
            output_column="answer",
            author_column="source",
            label_column="category"
        )
        backend = InMemoryBackend()
        
        loader.load(backend)
        
        sample = backend.get_item(0)
        self.assertEqual(sample.prompt, "Test question")
        self.assertEqual(sample.output, "Test answer")
        self.assertEqual(sample.author_id, "test_source")
        self.assertEqual(sample.label, "test_category")
        self.assertEqual(sample.metadata['metadata_field'], "metadata_value")

if __name__ == '__main__':
    unittest.main() 