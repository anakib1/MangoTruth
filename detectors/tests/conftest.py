"""Pytest configuration and shared fixtures for detectors tests."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any

from detectors.data.datasets import Dataset, TextSample
from detectors.data.datasets.interfaces import TextDatasetInterface
from detectors.training.interfaces import ITrainable, TrainingConfig
from detectors.training.configs import HuggingFaceTrainingConfig
from detectors.metrics import ClassificationMetrics, SplitConclusion, ClassificationRepresentations
from detectors.interfaces import IDetector


@pytest.fixture
def sample_text_samples():
    """Create sample text samples for testing."""
    return [
        TextSample(
            prompt="Write about AI",
            output="Artificial intelligence is transforming the world.",
            author_id="human_1",
            label="human"
        ),
        TextSample(
            prompt="Write about AI", 
            output="AI systems are revolutionizing multiple industries.",
            author_id="ai_1",
            label="ai"
        ),
        TextSample(
            prompt="Describe technology",
            output="Technology has become an integral part of our lives.",
            author_id="human_2", 
            label="human"
        ),
        TextSample(
            prompt="Describe technology",
            output="Technological advancement drives economic growth.",
            author_id="ai_2",
            label="ai"
        )
    ]


@pytest.fixture
def sample_dataset(sample_text_samples):
    """Create a sample dataset for testing."""
    return Dataset.from_samples(sample_text_samples)


@pytest.fixture
def mock_model():
    """Create a mock trainable model for testing."""
    model = Mock()
    model.get_labels.return_value = ["human", "ai"]
    model.get_num_parameters.return_value = 1000000
    model.get_num_trainable_parameters.return_value = 1000000
    model.predict_proba.return_value = np.array([0.7, 0.3])
    model.batch_predict.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
    model.store_weights.return_value = b"fake_weights"
    model.load_weights.return_value = None
    model.train_mode.return_value = None
    model.save_checkpoint.return_value = None
    model.load_checkpoint.return_value = None
    return model


@pytest.fixture
def mock_detector():
    """Create a mock detector for testing."""
    detector = Mock()
    detector.get_labels.return_value = ["human", "ai"]
    detector.get_num_parameters.return_value = 1000000
    detector.predict_proba.return_value = np.array([0.7, 0.3])
    
    # More realistic batch_predict that respects the input size
    def mock_batch_predict(texts):
        # Return predictions matching the input size
        batch_size = len(texts)
        return np.array([[0.6, 0.4]] * batch_size)
    
    detector.batch_predict = mock_batch_predict
    detector.__class__.__name__ = "MockDetector"  # For string formatting
    return detector


@pytest.fixture
def basic_training_config():
    """Create a basic training configuration for testing."""
    return TrainingConfig(
        epochs=3,
        batch_size=32,
        learning_rate=5e-5,
        output_dir="./test_output",
        run_id=uuid4()
    )


@pytest.fixture
def huggingface_training_config():
    """Create a HuggingFace training configuration for testing."""
    return HuggingFaceTrainingConfig(
        model_name="bert-base-uncased",
        epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        output_dir="./test_output",
        run_id=uuid4()
    )


@pytest.fixture
def sample_metrics():
    """Create sample classification metrics for testing."""
    return ClassificationMetrics(
        accuracy=0.85,
        precision=0.82,
        recall=0.88,
        f1=0.85,
        auc=0.90,
        tpr_at_1_percent_fpr=0.75,
        tpr_at_10_percent_fpr=0.85
    )


@pytest.fixture
def sample_conclusion(sample_metrics):
    """Create a sample conclusion for testing."""
    return SplitConclusion(
        metrics=sample_metrics,
        representations=ClassificationRepresentations(
            roc_curve=None,
            clf_report=None
        )
    )


class MockHuggingFaceDataset:
    """Custom mock class for HuggingFace datasets that properly supports magic methods."""
    
    def __init__(self, data=None, length=1000):
        self.data = data or [
            {"output": "Sample text 1", "label": 0, "prompt": "Write something", "user_id": "user1"},
            {"output": "Sample text 2", "label": 3, "prompt": "Write something", "user_id": "user2"}
        ]
        self._length = length
        
        # Mock methods
        self.filter = Mock(return_value=self)
        self.shuffle = Mock(return_value=self)
        self.select = Mock(return_value=self)
        
        # Mock pandas operations
        self._setup_pandas_mock()
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        return iter(self.data)
    
    def _setup_pandas_mock(self):
        """Setup pandas-related mocks for balancing operations."""
        mock_df = Mock()
        
        # Mock the pandas dataframe column access
        mock_label_series = Mock()
        mock_label_series.value_counts.return_value = Mock()
        mock_label_series.value_counts.return_value.min.return_value = 50
        mock_df.__getitem__ = Mock(return_value=mock_label_series)
        
        # Setup groupby mock chain
        mock_grouped = Mock()
        mock_applied = Mock()
        mock_index = Mock()
        mock_level_values = Mock()
        mock_level_values.values = [0, 1, 2, 3]
        mock_index.get_level_values.return_value = mock_level_values
        mock_applied.index = mock_index
        mock_grouped.apply.return_value = mock_applied
        mock_df.groupby.return_value = mock_grouped
        
        self.to_pandas = Mock(return_value=mock_df)


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset for testing."""
    return MockHuggingFaceDataset()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_neptune_nexus():
    """Create a mock Neptune nexus for testing."""
    nexus = Mock()
    nexus.conclude_run.return_value = None
    nexus.load_run_weights.return_value = b"fake_weights"
    return nexus


# Helper functions for tests
def create_mock_training_result(run_id=None, training_duration=100.0):
    """Create a mock training result for testing."""
    from detectors.training.interfaces import TrainingResult
    
    return TrainingResult(
        run_id=run_id or uuid4(),
        model_name="test_model",
        trainer_type="TestTrainer",
        start_time=datetime.now(),
        end_time=datetime.now(),
        training_duration_seconds=training_duration,
        train_conclusion=Mock(),
        model_weights=b"fake_weights",
        validation_conclusion=None,
        test_conclusion=None,
        best_checkpoint_path=None,
        training_history={},
        config=None,
        metadata={}
    )


def create_binary_classification_data(n_samples=100):
    """Create sample binary classification data for testing."""
    np.random.seed(42)
    true_labels = np.random.randint(0, 2, n_samples)
    predictions = np.random.rand(n_samples)
    # Make predictions somewhat correlated with true labels
    predictions = predictions * 0.6 + true_labels * 0.4
    return true_labels, predictions 