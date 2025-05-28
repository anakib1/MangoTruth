import unittest
from detectors.data.datasets.array_dataset import ArrayDataset
from detectors.data.datasets.base import TextSample

class TestArrayDataset(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.sample1 = TextSample(
            prompt="Test prompt 1",
            output="Test output 1",
            author_id="test_author",
            label="test_label",
            metadata={"test_key": "test_value"}
        )
        self.sample2 = TextSample(
            prompt="Test prompt 2",
            output="Test output 2",
            author_id="test_author",
            label="test_label"
        )
        
    def test_init_empty(self):
        """Test initialization with no samples."""
        dataset = ArrayDataset()
        self.assertEqual(len(dataset.samples), 0)
        
    def test_init_with_samples(self):
        """Test initialization with provided samples."""
        samples = [self.sample1, self.sample2]
        dataset = ArrayDataset(samples=samples)
        self.assertEqual(len(dataset.samples), 2)
        self.assertEqual(dataset.samples[0], self.sample1)
        self.assertEqual(dataset.samples[1], self.sample2)
        
    def test_load(self):
        """Test load method (should be a no-op)."""
        dataset = ArrayDataset([self.sample1])
        dataset.load()  # Should not raise any errors
        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0], self.sample1)
        
    def test_save(self):
        """Test save method (should be a no-op)."""
        dataset = ArrayDataset([self.sample1])
        dataset.save()  # Should not raise any errors
        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0], self.sample1)
        
    def test_modify_samples(self):
        """Test that samples can be modified after initialization."""
        dataset = ArrayDataset()
        dataset.samples.append(self.sample1)
        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0], self.sample1)
        
        dataset.samples.append(self.sample2)
        self.assertEqual(len(dataset.samples), 2)
        self.assertEqual(dataset.samples[1], self.sample2)
        
    def test_sample_attributes(self):
        """Test that sample attributes are preserved."""
        dataset = ArrayDataset([self.sample1])
        sample = dataset.samples[0]
        
        self.assertEqual(sample.prompt, "Test prompt 1")
        self.assertEqual(sample.output, "Test output 1")
        self.assertEqual(sample.author_id, "test_author")
        self.assertEqual(sample.label, "test_label")
        self.assertEqual(sample.metadata, {"test_key": "test_value"})
        
if __name__ == '__main__':
    unittest.main() 