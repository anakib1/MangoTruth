"""Unit tests for the new Dataset class.

This module contains comprehensive tests for the new Dataset class,
including backend operations, filtering, mapping, and dataset manipulation.
"""

import unittest
import tempfile
import os
from typing import List
from detectors.data.datasets import (
    Dataset, TextSample, InMemoryBackend, PandasBackend
)

class TestDataset(unittest.TestCase):
    """Test suite for the Dataset class."""
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.sample1 = TextSample(
            prompt="What is the capital of France?",
            output="The capital of France is Paris.",
            author_id="human",
            label="human",
            metadata={"category": "geography"}
        )
        self.sample2 = TextSample(
            prompt="Explain quantum computing",
            output="Quantum computing uses quantum mechanical phenomena...",
            author_id="gpt-4",
            label="ai",
            metadata={"category": "science"}
        )
        self.sample3 = TextSample(
            prompt="How do you make coffee?",
            output="To make coffee, you need...",
            author_id="human",
            label="human",
            metadata={"category": "cooking"}
        )
        self.samples = [self.sample1, self.sample2, self.sample3]
    
    def test_init_empty(self):
        """Test initialization of empty dataset."""
        dataset = Dataset()
        self.assertEqual(len(dataset), 0)
        self.assertIsInstance(dataset.backend, InMemoryBackend)
    
    def test_init_with_backend(self):
        """Test initialization with specific backend."""
        backend = PandasBackend()
        dataset = Dataset(backend=backend)
        self.assertEqual(len(dataset), 0)
        self.assertIsInstance(dataset.backend, PandasBackend)
    
    def test_from_samples(self):
        """Test creating dataset from samples."""
        dataset = Dataset.from_samples(self.samples)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], self.sample1)
        self.assertEqual(dataset[1], self.sample2)
        self.assertEqual(dataset[2], self.sample3)
    
    def test_getitem_single(self):
        """Test getting single item by index."""
        dataset = Dataset.from_samples(self.samples)
        
        # Test positive indices
        self.assertEqual(dataset[0], self.sample1)
        self.assertEqual(dataset[1], self.sample2)
        self.assertEqual(dataset[2], self.sample3)
        
        # Test negative indices
        self.assertEqual(dataset[-1], self.sample3)
        self.assertEqual(dataset[-2], self.sample2)
        self.assertEqual(dataset[-3], self.sample1)
    
    def test_getitem_slice(self):
        """Test getting slice of items."""
        dataset = Dataset.from_samples(self.samples)
        
        subset = dataset[1:3]
        self.assertIsInstance(subset, Dataset)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0], self.sample2)
        self.assertEqual(subset[1], self.sample3)
        
        # Test slice with step
        subset = dataset[::2]
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0], self.sample1)
        self.assertEqual(subset[1], self.sample3)
    
    def test_getitem_sequence(self):
        """Test getting items by sequence of indices."""
        dataset = Dataset.from_samples(self.samples)
        
        subset = dataset[[0, 2]]
        self.assertIsInstance(subset, Dataset)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0], self.sample1)
        self.assertEqual(subset[1], self.sample3)
    
    def test_iteration(self):
        """Test iteration over dataset."""
        dataset = Dataset.from_samples(self.samples)
        
        samples = list(dataset)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0], self.sample1)
        self.assertEqual(samples[1], self.sample2)
        self.assertEqual(samples[2], self.sample3)
    
    def test_filter(self):
        """Test filtering dataset."""
        dataset = Dataset.from_samples(self.samples)
        
        # Filter by label
        human_samples = dataset.filter(lambda s: s.label == "human")
        self.assertEqual(len(human_samples), 2)
        self.assertEqual(human_samples[0], self.sample1)
        self.assertEqual(human_samples[1], self.sample3)
        
        # Filter by metadata
        geography_samples = dataset.filter(
            lambda s: s.metadata.get("category") == "geography"
        )
        self.assertEqual(len(geography_samples), 1)
        self.assertEqual(geography_samples[0], self.sample1)
    
    def test_map(self):
        """Test mapping function over dataset."""
        dataset = Dataset.from_samples(self.samples)
        
        # Map to uppercase prompts
        mapped = dataset.map(
            lambda s: TextSample(
                s.prompt.upper(),
                s.output,
                s.author_id,
                s.label,
                s.metadata
            )
        )
        
        self.assertEqual(len(mapped), 3)
        self.assertEqual(mapped[0].prompt, "WHAT IS THE CAPITAL OF FRANCE?")
        self.assertEqual(mapped[0].output, self.sample1.output)
    
    def test_batch(self):
        """Test batch iteration."""
        dataset = Dataset.from_samples(self.samples)
        
        batches = list(dataset.batch(2))
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 1)
        
        # Test exact batch size
        batches = list(dataset.batch(3))
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)
    
    def test_shuffle(self):
        """Test dataset shuffling."""
        dataset = Dataset.from_samples(self.samples)
        
        # Test with seed for reproducibility
        shuffled1 = dataset.shuffle(seed=42)
        shuffled2 = dataset.shuffle(seed=42)
        
        # Should be same order with same seed
        self.assertEqual(len(shuffled1), len(shuffled2))
        for i in range(len(shuffled1)):
            self.assertEqual(shuffled1[i], shuffled2[i])
        
        # Original should be unchanged
        self.assertEqual(dataset[0], self.sample1)
    
    def test_split(self):
        """Test dataset splitting."""
        # Create larger dataset for meaningful splits
        samples = [
            TextSample(f"Q{i}", f"A{i}", f"author{i%2}", f"label{i%2}")
            for i in range(10)
        ]
        dataset = Dataset.from_samples(samples)
        
        splits = dataset.split([0.6, 0.2, 0.2], seed=42)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[0]), 6)  # 60%
        self.assertEqual(len(splits[1]), 2)  # 20%
        self.assertEqual(len(splits[2]), 2)  # 20%
        
        # Test ratios don't sum to 1
        with self.assertRaises(ValueError):
            dataset.split([0.5, 0.3, 0.3])
    
    def test_get_by_label(self):
        """Test filtering by label."""
        dataset = Dataset.from_samples(self.samples)
        
        human_dataset = dataset.get_by_label("human")
        self.assertEqual(len(human_dataset), 2)
        for sample in human_dataset:
            self.assertEqual(sample.label, "human")
        
        ai_dataset = dataset.get_by_label("ai")
        self.assertEqual(len(ai_dataset), 1)
        self.assertEqual(ai_dataset[0], self.sample2)
    
    def test_get_by_author(self):
        """Test filtering by author."""
        dataset = Dataset.from_samples(self.samples)
        
        human_author = dataset.get_by_author("human")
        self.assertEqual(len(human_author), 2)
        for sample in human_author:
            self.assertEqual(sample.author_id, "human")
        
        gpt4_author = dataset.get_by_author("gpt-4")
        self.assertEqual(len(gpt4_author), 1)
        self.assertEqual(gpt4_author[0], self.sample2)
    
    def test_get_labels(self):
        """Test getting unique labels."""
        dataset = Dataset.from_samples(self.samples)
        
        labels = dataset.get_labels()
        self.assertEqual(sorted(labels), ["ai", "human"])
    
    def test_get_authors(self):
        """Test getting unique authors."""
        dataset = Dataset.from_samples(self.samples)
        
        authors = dataset.get_authors()
        self.assertEqual(sorted(authors), ["gpt-4", "human"])
    
    def test_add_sample(self):
        """Test adding single sample."""
        dataset = Dataset()
        self.assertEqual(len(dataset), 0)
        
        dataset.add_sample(self.sample1)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0], self.sample1)
    
    def test_add_samples(self):
        """Test adding multiple samples."""
        dataset = Dataset()
        self.assertEqual(len(dataset), 0)
        
        dataset.add_samples(self.samples)
        self.assertEqual(len(dataset), 3)
        for i, sample in enumerate(self.samples):
            self.assertEqual(dataset[i], sample)
    
    def test_save_load(self):
        """Test saving and loading dataset."""
        dataset = Dataset.from_samples(self.samples)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # Save dataset
            dataset.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load into new dataset
            new_dataset = Dataset()
            new_dataset.load(temp_path)
            
            # Verify content
            self.assertEqual(len(new_dataset), len(dataset))
            for i in range(len(dataset)):
                original = dataset[i]
                loaded = new_dataset[i]
                self.assertEqual(original.prompt, loaded.prompt)
                self.assertEqual(original.output, loaded.output)
                self.assertEqual(original.author_id, loaded.author_id)
                self.assertEqual(original.label, loaded.label)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        dataset = Dataset.from_samples(self.samples)
        usage = dataset.get_memory_usage()
        self.assertIsInstance(usage, int)
        self.assertGreater(usage, 0)
    
    def test_to_list(self):
        """Test converting to list."""
        dataset = Dataset.from_samples(self.samples)
        samples_list = dataset.to_list()
        
        self.assertIsInstance(samples_list, list)
        self.assertEqual(len(samples_list), len(self.samples))
        for i, sample in enumerate(self.samples):
            self.assertEqual(samples_list[i], sample)
    
    def test_repr(self):
        """Test string representation."""
        dataset = Dataset.from_samples(self.samples)
        repr_str = repr(dataset)
        
        self.assertIn("Dataset", repr_str)
        self.assertIn("size=3", repr_str)
        self.assertIn("InMemoryBackend", repr_str)
        self.assertIn("labels=", repr_str)

if __name__ == '__main__':
    unittest.main() 