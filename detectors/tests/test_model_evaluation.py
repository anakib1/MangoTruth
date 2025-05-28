import unittest
import numpy as np
from detectors.models.zoo import ModelRegistry, HuggingFaceConfig, HuggingFaceDetector, evaluate
from detectors.models.zoo.const_detector import ConstDetector
from detectors.interfaces import IDetector
from detectors.data.datasets.array_dataset import ArrayDataset
from detectors.data.datasets.base import TextSample
from typing import List
import torch


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # Create test data with more samples for better metrics
        self.samples = [
            TextSample(
                prompt="",
                output=f"Test text {i}",
                author_id="test_author",
                label=label,
                metadata={}
            )
            for i, label in enumerate(["human", "ai", "human", "ai", "human"])
        ]
        self.test_dataset = ArrayDataset(samples=self.samples)
        
    def test_evaluation_with_const_detector(self):
        """Test evaluation with a constant detector"""
        detector = ConstDetector(human_prob=0.8)
        results = evaluate(detector, self.test_dataset)
        
        # Verify results structure
        self.assertIsNotNone(results)
        self.assertTrue(hasattr(results, 'metrics'))
        
        # Check metrics
        metrics = results.metrics
        self.assertIsNotNone(metrics.accuracy)
        self.assertIsNotNone(metrics.precision)
        self.assertIsNotNone(metrics.recall)
        self.assertIsNotNone(metrics.f1)
        
        # Verify metrics are reasonable
        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)
        
    def test_model_registry(self):
        """Test model registry functionality"""
        registry = ModelRegistry()
        
        # Test registering and getting a model
        registry.register_model("const", ConstDetector)
        model = registry.get_model("const", human_prob=0.8)
        
        self.assertIsInstance(model, ConstDetector)
        self.assertEqual(model.human_prob, 0.8)
        
        # Test listing models
        models = registry.list_models()
        self.assertIn("const", models)
        
    def test_huggingface_detector_interface(self):
        """Test that HuggingFace detector implements the interface correctly"""
        config = HuggingFaceConfig(
            model_name="bert-base-uncased",
            num_labels=2,
            max_length=512
        )
        
        detector = HuggingFaceDetector(config)
        
        # Test interface methods
        self.assertIsInstance(detector.get_labels(), list)
        self.assertEqual(len(detector.get_labels()), 2)
        
        # Test prediction methods
        pred = detector.predict_proba("Test text")
        self.assertEqual(pred.shape, (2,))
        self.assertAlmostEqual(pred.sum(), 1.0)
        
        batch_preds = detector.batch_predict(["Test 1", "Test 2"])
        self.assertEqual(batch_preds.shape, (2, 2))
        self.assertTrue(np.allclose(batch_preds.sum(axis=1), 1.0))
        
    def test_evaluation_metrics(self):
        """Test that evaluation produces correct metrics"""
        # Create a detector with specific human probability
        detector = ConstDetector(human_prob=0.7)
        
        results = evaluate(detector, self.test_dataset)
        
        # Verify metrics
        self.assertIsNotNone(results.metrics)
        metrics = results.metrics
        self.assertTrue(hasattr(metrics, 'accuracy'))
        self.assertTrue(hasattr(metrics, 'precision'))
        self.assertTrue(hasattr(metrics, 'recall'))
        self.assertTrue(hasattr(metrics, 'f1'))
        
        # Verify metrics are reasonable
        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)
        self.assertGreaterEqual(metrics.precision, 0)
        self.assertLessEqual(metrics.precision, 1)
        self.assertGreaterEqual(metrics.recall, 0)
        self.assertLessEqual(metrics.recall, 1)
        self.assertGreaterEqual(metrics.f1, 0)
        self.assertLessEqual(metrics.f1, 1)


if __name__ == '__main__':
    unittest.main() 