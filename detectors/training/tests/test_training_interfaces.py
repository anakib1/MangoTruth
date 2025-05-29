"""Unit tests for training interfaces and configurations.

This module tests the core training infrastructure components.
"""

import unittest
from uuid import UUID
from typing import Dict, Any, Tuple

from detectors.training.interfaces import TrainingConfig, TrainingResult, ITrainable
from detectors.training.configs import BaseTrainingConfig, HuggingFaceTrainingConfig


class MockTrainable(ITrainable):
    """Mock implementation of ITrainable for testing."""
    
    def __init__(self):
        self.is_training = False
        self.parameters = {"weight": 1.0, "bias": 0.0}
    
    def train_mode(self, mode: bool = True) -> None:
        self.is_training = mode
    
    def get_trainable_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
    
    def compute_loss(self, inputs: Dict[str, Any], labels: Any) -> Tuple[Any, Dict[str, Any]]:
        # Simple mock loss computation
        return 0.5, {"logits": [0.1, 0.9]}
    
    def save_checkpoint(self, path: str) -> None:
        # Mock save
        pass
    
    def load_checkpoint(self, path: str) -> None:
        # Mock load
        pass
    
    def predict_proba(self, text: str):
        return [0.3, 0.7]
    
    def get_labels(self):
        return ["human", "ai"]
    
    def store_weights(self) -> bytes:
        return b"mock_weights"
    
    def load_weights(self, weights: bytes) -> None:
        pass


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig classes."""
    
    def test_base_training_config_creation(self):
        """Test creating a base training configuration."""
        config = BaseTrainingConfig(
            epochs=5,
            batch_size=16,
            learning_rate=1e-4
        )
        
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertIsInstance(config.run_id, UUID)
    
    def test_base_training_config_serialization(self):
        """Test serialization of base training configuration."""
        config = BaseTrainingConfig(epochs=10)
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["epochs"], 10)
        self.assertIsInstance(config_dict["run_id"], str)
        
        # Test from_dict
        loaded_config = BaseTrainingConfig.from_dict(config_dict)
        self.assertEqual(loaded_config.epochs, config.epochs)
        self.assertEqual(loaded_config.run_id, config.run_id)
    
    def test_huggingface_training_config(self):
        """Test HuggingFace training configuration."""
        config = HuggingFaceTrainingConfig(
            model_name="bert-base-uncased",
            num_labels=3,
            epochs=2,
            fp16=True
        )
        
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.tokenizer_name, "bert-base-uncased")  # Should default to model_name
        self.assertEqual(config.num_labels, 3)
        self.assertTrue(config.fp16)
        
        # Test HF args conversion
        hf_args = config.to_hf_training_args()
        self.assertIsInstance(hf_args, dict)
        self.assertEqual(hf_args["num_train_epochs"], 2)
        self.assertTrue(hf_args["fp16"])
    
    def test_huggingface_config_batch_size_defaults(self):
        """Test that per-device batch sizes default correctly."""
        config = HuggingFaceTrainingConfig(batch_size=8)
        
        self.assertEqual(config.per_device_train_batch_size, 8)
        self.assertEqual(config.per_device_eval_batch_size, 8)


class TestTrainingInterfaces(unittest.TestCase):
    """Test training interfaces."""
    
    def test_trainable_interface(self):
        """Test the ITrainable interface implementation."""
        model = MockTrainable()
        
        # Test train mode
        self.assertFalse(model.is_training)
        model.train_mode(True)
        self.assertTrue(model.is_training)
        model.train_mode(False)
        self.assertFalse(model.is_training)
        
        # Test parameters
        params = model.get_trainable_parameters()
        self.assertEqual(params["weight"], 1.0)
        self.assertEqual(params["bias"], 0.0)
        
        # Test parameter update
        model.update_parameters({"weight": 2.0})
        params = model.get_trainable_parameters()
        self.assertEqual(params["weight"], 2.0)
        
        # Test loss computation
        loss, aux = model.compute_loss({"input": "test"}, [1])
        self.assertEqual(loss, 0.5)
        self.assertIn("logits", aux)


class TestTrainingResult(unittest.TestCase):
    """Test TrainingResult dataclass."""
    
    def test_training_result_creation(self):
        """Test creating a TrainingResult."""
        from datetime import datetime
        from detectors.metrics import Conclusion, SplitConclusion, ClassificationMetrics
        
        # Create mock metrics
        metrics = ClassificationMetrics(
            tpr_at_1_percent_fpr=0.9,
            tpr_at_10_percent_fpr=0.95,
            auc=0.98,
            f1=0.85,
            accuracy=0.9,
            precision=0.88,
            recall=0.82
        )
        
        split_conclusion = SplitConclusion(
            metrics=metrics,
            representations=None  # Skip for unit test
        )
        
        conclusion = Conclusion(
            weights=b"mock_weights",
            detector_handle="test_model",
            datasets=["test_dataset"],
            train_conclusion=split_conclusion,
            validation_conclusion=split_conclusion
        )
        
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = TrainingResult(
            run_id=UUID('12345678-1234-5678-1234-567812345678'),
            model_name="test_model",
            trainer_type="TestTrainer",
            start_time=start_time,
            end_time=end_time,
            training_duration_seconds=60.0,
            train_conclusion=conclusion,
            model_weights=b"model_weights"
        )
        
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.trainer_type, "TestTrainer")
        self.assertEqual(result.training_duration_seconds, 60.0)
        self.assertIsInstance(result.training_history, dict)


if __name__ == "__main__":
    unittest.main() 