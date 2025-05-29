"""Unit tests for dataset storage backends.

This module contains tests for the storage backend implementations,
including InMemoryBackend and PandasBackend.
"""

import unittest
import tempfile
import os
import numpy as np
from detectors.data.datasets.backends import InMemoryBackend, PandasBackend
from detectors.data.datasets.interfaces import TextSample

class TestInMemoryBackend(unittest.TestCase):
    """Test suite for InMemoryBackend."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample1 = TextSample("Q1", "A1", "author1", "human")
        self.sample2 = TextSample("Q2", "A2", "author2", "ai")
        self.sample3 = TextSample("Q3", "A3", "author3", "human")
        self.samples = [self.sample1, self.sample2, self.sample3]
    
    def test_init_empty(self):
        """Test initialization with no data."""
        backend = InMemoryBackend()
        self.assertEqual(len(backend), 0)
        self.assertEqual(backend.data, [])
    
    def test_init_with_data(self):
        """Test initialization with data."""
        backend = InMemoryBackend(self.samples)
        self.assertEqual(len(backend), 3)
        self.assertEqual(backend.data, self.samples)
    
    def test_get_item(self):
        """Test getting single item."""
        backend = InMemoryBackend(self.samples)
        
        # Test positive indices
        self.assertEqual(backend.get_item(0), self.sample1)
        self.assertEqual(backend.get_item(1), self.sample2)
        self.assertEqual(backend.get_item(2), self.sample3)
        
        # Test negative indices
        self.assertEqual(backend.get_item(-1), self.sample3)
        self.assertEqual(backend.get_item(-2), self.sample2)
        
        # Test out of range
        with self.assertRaises(IndexError):
            backend.get_item(3)
        with self.assertRaises(IndexError):
            backend.get_item(-4)
    
    def test_get_items_slice(self):
        """Test getting items by slice."""
        backend = InMemoryBackend(self.samples)
        
        items = backend.get_items(slice(0, 2))
        self.assertEqual(items, [self.sample1, self.sample2])
        
        items = backend.get_items(slice(1, None))
        self.assertEqual(items, [self.sample2, self.sample3])
        
        items = backend.get_items(slice(None, None, 2))
        self.assertEqual(items, [self.sample1, self.sample3])
    
    def test_get_items_sequence(self):
        """Test getting items by sequence of indices."""
        backend = InMemoryBackend(self.samples)
        
        items = backend.get_items([0, 2])
        self.assertEqual(items, [self.sample1, self.sample3])
        
        items = backend.get_items([2, 1, 0])
        self.assertEqual(items, [self.sample3, self.sample2, self.sample1])
    
    def test_add_item(self):
        """Test adding single item."""
        backend = InMemoryBackend()
        self.assertEqual(len(backend), 0)
        
        backend.add_item(self.sample1)
        self.assertEqual(len(backend), 1)
        self.assertEqual(backend.get_item(0), self.sample1)
    
    def test_add_items(self):
        """Test adding multiple items."""
        backend = InMemoryBackend()
        self.assertEqual(len(backend), 0)
        
        backend.add_items(self.samples)
        self.assertEqual(len(backend), 3)
        for i, sample in enumerate(self.samples):
            self.assertEqual(backend.get_item(i), sample)
    
    def test_delete_item(self):
        """Test deleting item."""
        backend = InMemoryBackend(self.samples.copy())
        self.assertEqual(len(backend), 3)
        
        backend.delete_item(1)
        self.assertEqual(len(backend), 2)
        self.assertEqual(backend.get_item(0), self.sample1)
        self.assertEqual(backend.get_item(1), self.sample3)
    
    def test_clear(self):
        """Test clearing all items."""
        backend = InMemoryBackend(self.samples.copy())
        self.assertEqual(len(backend), 3)
        
        backend.clear()
        self.assertEqual(len(backend), 0)
        self.assertEqual(backend.data, [])
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        backend = InMemoryBackend(self.samples)
        usage = backend.get_memory_usage()
        self.assertIsInstance(usage, int)
        self.assertGreater(usage, 0)
    
    def test_to_list(self):
        """Test converting to list."""
        backend = InMemoryBackend(self.samples)
        items_list = backend.to_list()
        
        self.assertIsInstance(items_list, list)
        self.assertEqual(items_list, self.samples)
        # Ensure it's a copy
        self.assertIsNot(items_list, backend.data)
    
    def test_save_load(self):
        """Test saving and loading."""
        backend = InMemoryBackend(self.samples)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # Save
            backend.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load into new backend
            new_backend = InMemoryBackend()
            new_backend.load(temp_path)
            
            # Verify
            self.assertEqual(len(new_backend), len(backend))
            for i in range(len(backend)):
                self.assertEqual(new_backend.get_item(i), backend.get_item(i))
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_copy(self):
        """Test copying backend."""
        backend = InMemoryBackend(self.samples)
        copied = backend.copy()
        
        self.assertIsInstance(copied, InMemoryBackend)
        self.assertEqual(len(copied), len(backend))
        for i in range(len(backend)):
            self.assertEqual(copied.get_item(i), backend.get_item(i))
        
        # Ensure it's a deep copy
        self.assertIsNot(copied.data, backend.data)
    
    def test_get_indices(self):
        """Test getting indices array."""
        backend = InMemoryBackend(self.samples)
        indices = backend.get_indices()
        
        self.assertIsInstance(indices, np.ndarray)
        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))

class TestPandasBackend(unittest.TestCase):
    """Test suite for PandasBackend."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample1 = TextSample("Q1", "A1", "author1", "human", {"key1": "value1"})
        self.sample2 = TextSample("Q2", "A2", "author2", "ai", {"key2": "value2"})
        self.sample3 = TextSample("Q3", "A3", "author3", "human", {})
        self.samples = [self.sample1, self.sample2, self.sample3]
    
    def test_init_empty(self):
        """Test initialization with no data."""
        backend = PandasBackend()
        self.assertEqual(len(backend), 0)
        expected_columns = ['prompt', 'output', 'author_id', 'label', 'metadata']
        self.assertEqual(list(backend.df.columns), expected_columns)
    
    def test_get_item(self):
        """Test getting single item."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        
        item = backend.get_item(0)
        self.assertEqual(item.prompt, "Q1")
        self.assertEqual(item.output, "A1")
        self.assertEqual(item.author_id, "author1")
        self.assertEqual(item.label, "human")
        self.assertEqual(item.metadata, {"key1": "value1"})
        
        # Test negative indices
        item = backend.get_item(-1)
        self.assertEqual(item.prompt, "Q3")
        
        # Test out of range
        with self.assertRaises(IndexError):
            backend.get_item(3)
    
    def test_get_items(self):
        """Test getting multiple items."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        
        # Test slice
        items = backend.get_items(slice(0, 2))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].prompt, "Q1")
        self.assertEqual(items[1].prompt, "Q2")
        
        # Test sequence
        items = backend.get_items([0, 2])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].prompt, "Q1")
        self.assertEqual(items[1].prompt, "Q3")
    
    def test_add_item(self):
        """Test adding single item."""
        backend = PandasBackend()
        self.assertEqual(len(backend), 0)
        
        backend.add_item(self.sample1)
        self.assertEqual(len(backend), 1)
        
        item = backend.get_item(0)
        self.assertEqual(item.prompt, "Q1")
    
    def test_add_items(self):
        """Test adding multiple items."""
        backend = PandasBackend()
        self.assertEqual(len(backend), 0)
        
        backend.add_items(self.samples)
        self.assertEqual(len(backend), 3)
        
        for i, expected in enumerate(self.samples):
            item = backend.get_item(i)
            self.assertEqual(item.prompt, expected.prompt)
            self.assertEqual(item.output, expected.output)
            self.assertEqual(item.author_id, expected.author_id)
            self.assertEqual(item.label, expected.label)
    
    def test_delete_item(self):
        """Test deleting item."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        self.assertEqual(len(backend), 3)
        
        backend.delete_item(1)
        self.assertEqual(len(backend), 2)
        
        # Check remaining items
        self.assertEqual(backend.get_item(0).prompt, "Q1")
        self.assertEqual(backend.get_item(1).prompt, "Q3")
    
    def test_clear(self):
        """Test clearing all items."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        self.assertEqual(len(backend), 3)
        
        backend.clear()
        self.assertEqual(len(backend), 0)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        usage = backend.get_memory_usage()
        self.assertIsInstance(usage, (int, np.integer))
        self.assertGreater(usage, 0)
    
    def test_to_list(self):
        """Test converting to list."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        
        items_list = backend.to_list()
        self.assertIsInstance(items_list, list)
        self.assertEqual(len(items_list), 3)
        
        for i, expected in enumerate(self.samples):
            item = items_list[i]
            self.assertEqual(item.prompt, expected.prompt)
            self.assertEqual(item.output, expected.output)
    
    def test_copy(self):
        """Test copying backend."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        
        copied = backend.copy()
        self.assertIsInstance(copied, PandasBackend)
        self.assertEqual(len(copied), len(backend))
        
        # Verify content
        for i in range(len(backend)):
            original = backend.get_item(i)
            copied_item = copied.get_item(i)
            self.assertEqual(original.prompt, copied_item.prompt)
        
        # Ensure it's a copy
        self.assertIsNot(copied.df, backend.df)
    
    def test_get_indices(self):
        """Test getting indices array."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        
        indices = backend.get_indices()
        self.assertIsInstance(indices, np.ndarray)
        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))
    
    def test_pandas_specific_methods(self):
        """Test pandas-specific methods."""
        backend = PandasBackend()
        backend.add_items(self.samples)
        
        # Test label counts
        label_counts = backend.get_label_counts()
        self.assertEqual(label_counts['human'], 2)
        self.assertEqual(label_counts['ai'], 1)
        
        # Test author counts
        author_counts = backend.get_author_counts()
        self.assertEqual(author_counts['author1'], 1)
        self.assertEqual(author_counts['author2'], 1)
        self.assertEqual(author_counts['author3'], 1)
        
        # Test query
        human_backend = backend.query("label == 'human'")
        self.assertIsInstance(human_backend, PandasBackend)
        self.assertEqual(len(human_backend), 2)
        
        # Test groupby
        grouped = backend.groupby('label')
        self.assertEqual(len(grouped), 2)  # Two groups: human and ai

if __name__ == '__main__':
    unittest.main() 