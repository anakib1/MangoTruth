"""Unit tests for dataset factory functions.

This module contains tests for the factory functions that create
and manipulate datasets with different backends and loaders.
"""

import unittest
from unittest.mock import Mock, patch
from detectors.data.datasets import (
    create_dataset, convert_backend, merge_datasets,
    Dataset, TextSample, InMemoryBackend, PandasBackend
)

class TestFactoryFunctions(unittest.TestCase):
    """Test suite for factory functions."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample1 = TextSample("Q1", "A1", "author1", "human")
        self.sample2 = TextSample("Q2", "A2", "author2", "ai")
        self.sample3 = TextSample("Q3", "A3", "author3", "human")
        self.samples = [self.sample1, self.sample2, self.sample3]
    
    def test_create_dataset_memory(self):
        """Test creating dataset with memory backend."""
        dataset = create_dataset("memory")
        
        self.assertIsInstance(dataset, Dataset)
        self.assertIsInstance(dataset.backend, InMemoryBackend)
        self.assertEqual(len(dataset), 0)
    
    def test_create_dataset_pandas(self):
        """Test creating dataset with pandas backend."""
        dataset = create_dataset("pandas")
        
        self.assertIsInstance(dataset, Dataset)
        self.assertIsInstance(dataset.backend, PandasBackend)
        self.assertEqual(len(dataset), 0)
    
    def test_create_dataset_invalid_backend(self):
        """Test creating dataset with invalid backend type."""
        with self.assertRaises(ValueError) as context:
            create_dataset("invalid_backend")
        
        self.assertIn("Unknown backend type", str(context.exception))
    
    def test_convert_backend_memory_to_pandas(self):
        """Test converting from memory to pandas backend."""
        # Create memory dataset
        memory_dataset = Dataset.from_samples(self.samples)
        self.assertIsInstance(memory_dataset.backend, InMemoryBackend)
        
        # Convert to pandas
        pandas_dataset = convert_backend(memory_dataset, "pandas")
        
        self.assertIsInstance(pandas_dataset, Dataset)
        self.assertIsInstance(pandas_dataset.backend, PandasBackend)
        self.assertEqual(len(pandas_dataset), len(memory_dataset))
        
        # Verify content is preserved
        for i in range(len(memory_dataset)):
            original = memory_dataset[i]
            converted = pandas_dataset[i]
            self.assertEqual(original.prompt, converted.prompt)
            self.assertEqual(original.output, converted.output)
            self.assertEqual(original.author_id, converted.author_id)
            self.assertEqual(original.label, converted.label)
    
    def test_convert_backend_pandas_to_memory(self):
        """Test converting from pandas to memory backend."""
        # Create pandas dataset
        pandas_backend = PandasBackend()
        pandas_backend.add_items(self.samples)
        pandas_dataset = Dataset(backend=pandas_backend)
        
        # Convert to memory
        memory_dataset = convert_backend(pandas_dataset, "memory")
        
        self.assertIsInstance(memory_dataset, Dataset)
        self.assertIsInstance(memory_dataset.backend, InMemoryBackend)
        self.assertEqual(len(memory_dataset), len(pandas_dataset))
        
        # Verify content is preserved
        for i in range(len(pandas_dataset)):
            original = pandas_dataset[i]
            converted = memory_dataset[i]
            self.assertEqual(original.prompt, converted.prompt)
            self.assertEqual(original.output, converted.output)
            self.assertEqual(original.author_id, converted.author_id)
            self.assertEqual(original.label, converted.label)
    
    def test_convert_backend_invalid_type(self):
        """Test converting to invalid backend type."""
        dataset = Dataset.from_samples(self.samples)
        
        with self.assertRaises(ValueError) as context:
            convert_backend(dataset, "invalid_backend")
        
        self.assertIn("Unknown backend type", str(context.exception))
    
    def test_convert_backend_preserves_metadata(self):
        """Test that backend conversion preserves metadata."""
        dataset = Dataset.from_samples(self.samples)
        dataset.metadata = {"test_key": "test_value", "source": "test"}
        
        converted = convert_backend(dataset, "pandas")
        
        self.assertEqual(converted.metadata, dataset.metadata)
        self.assertIsNot(converted.metadata, dataset.metadata)  # Should be a copy
    
    def test_merge_datasets_default_backend(self):
        """Test merging datasets using default backend type."""
        # Create datasets with different backend types
        dataset1 = Dataset.from_samples([self.sample1])
        dataset2 = Dataset.from_samples([self.sample2])
        dataset3 = Dataset.from_samples([self.sample3])
        
        merged = merge_datasets([dataset1, dataset2, dataset3])
        
        self.assertIsInstance(merged, Dataset)
        self.assertEqual(len(merged), 3)
        self.assertIsInstance(merged.backend, InMemoryBackend)  # Default to first dataset's type
        
        # Verify content
        self.assertEqual(merged[0], self.sample1)
        self.assertEqual(merged[1], self.sample2)
        self.assertEqual(merged[2], self.sample3)
    
    def test_merge_datasets_explicit_backend(self):
        """Test merging datasets with explicit backend type."""
        dataset1 = Dataset.from_samples([self.sample1])
        dataset2 = Dataset.from_samples([self.sample2])
        
        merged = merge_datasets([dataset1, dataset2], backend_type="pandas")
        
        self.assertIsInstance(merged, Dataset)
        self.assertEqual(len(merged), 2)
        self.assertIsInstance(merged.backend, PandasBackend)
    
    def test_merge_datasets_empty_list(self):
        """Test merging empty list of datasets."""
        with self.assertRaises(ValueError) as context:
            merge_datasets([])
        
        self.assertIn("No datasets provided", str(context.exception))
    
    def test_merge_datasets_preserves_metadata(self):
        """Test that merging preserves metadata from source datasets."""
        dataset1 = Dataset.from_samples([self.sample1])
        dataset1.metadata = {"source": "dataset1", "version": 1}
        
        dataset2 = Dataset.from_samples([self.sample2])
        dataset2.metadata = {"source": "dataset2", "version": 2}
        
        merged = merge_datasets([dataset1, dataset2])
        
        self.assertIn("merged_from", merged.metadata)
        self.assertIn("num_sources", merged.metadata)
        self.assertEqual(merged.metadata["num_sources"], 2)
        self.assertEqual(len(merged.metadata["merged_from"]), 2)
    
    def test_merge_datasets_with_pandas_backend(self):
        """Test merging when first dataset has pandas backend."""
        # Create dataset with pandas backend
        pandas_backend = PandasBackend()
        pandas_backend.add_items([self.sample1])
        pandas_dataset = Dataset(backend=pandas_backend)
        
        memory_dataset = Dataset.from_samples([self.sample2])
        
        merged = merge_datasets([pandas_dataset, memory_dataset])
        
        self.assertIsInstance(merged.backend, PandasBackend)
        self.assertEqual(len(merged), 2)

if __name__ == '__main__':
    unittest.main() 