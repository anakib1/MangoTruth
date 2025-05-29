"""Tests for BaseTrainer class."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import numpy as np

from detectors.training.trainers.base_trainer import BaseTrainer
from detectors.training.interfaces import TrainingConfig, TrainingResult
from detectors.metrics import Conclusion, ClassificationMetrics, SplitConclusion
from detectors.evaluation import EvaluationResult


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""
    
    def train(self, train_dataset, validation_dataset=None, test_dataset=None):
        start_time = datetime.now()
        
        # Simulate training
        train_conclusion = Mock()
        val_conclusion = None
        test_conclusion = None
        
        if validation_dataset:
            val_conclusion = self.evaluate(validation_dataset)
        if test_dataset:
            test_conclusion = self.evaluate(test_dataset)
        
        end_time = datetime.now()
        
        return self._create_training_result(
            start_time=start_time,
            end_time=end_time,
            train_conclusion=train_conclusion,
            validation_conclusion=val_conclusion,
            test_conclusion=test_conclusion
        )


class TestBaseTrainer:
    """Tests for BaseTrainer class."""
    
    def test_initialization(self, mock_model, basic_training_config, tmp_path):
        """Test BaseTrainer initialization."""
        basic_training_config.output_dir = str(tmp_path / "output")
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        assert trainer.model == mock_model
        assert trainer.config == basic_training_config
        assert hasattr(trainer, '_training_history')
        assert hasattr(trainer, '_best_metrics')
        assert hasattr(trainer, '_evaluator')
        assert isinstance(trainer._training_history, dict)
        assert isinstance(trainer._best_metrics, dict)
    
    def test_setup_creates_output_directory(self, mock_model, basic_training_config, tmp_path):
        """Test that setup creates output directory and saves config."""
        output_dir = tmp_path / "test_output"
        basic_training_config.output_dir = str(output_dir)
        
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Check output directory was created
        assert output_dir.exists()
        
        # Check config was saved
        config_file = output_dir / "training_config.json"
        assert config_file.exists()
        
        # Verify config content
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["epochs"] == basic_training_config.epochs
        assert saved_config["batch_size"] == basic_training_config.batch_size
    
    def test_log_metrics(self, mock_model, basic_training_config):
        """Test logging metrics to training history."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Log some metrics
        metrics = {"accuracy": 0.85, "loss": 0.25, "f1": 0.82}
        trainer._log_metrics(metrics, step=1, prefix="train_")
        
        # Check training history
        assert "train_accuracy" in trainer._training_history
        assert "train_loss" in trainer._training_history
        assert "train_f1" in trainer._training_history
        
        assert trainer._training_history["train_accuracy"] == [0.85]
        assert trainer._training_history["train_loss"] == [0.25]
        assert trainer._training_history["train_f1"] == [0.82]
        
        # Check best metrics (accuracy and f1 should be tracked as higher is better)
        assert trainer._best_metrics["train_accuracy"] == 0.85
        assert trainer._best_metrics["train_f1"] == 0.82
        # Loss should be tracked as lower is better
        assert trainer._best_metrics["train_loss"] == 0.25
    
    def test_log_metrics_multiple_steps(self, mock_model, basic_training_config):
        """Test logging metrics across multiple steps."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Log metrics for step 1
        trainer._log_metrics({"accuracy": 0.80, "loss": 0.30}, step=1)
        
        # Log metrics for step 2
        trainer._log_metrics({"accuracy": 0.85, "loss": 0.25}, step=2)
        
        # Log metrics for step 3
        trainer._log_metrics({"accuracy": 0.82, "loss": 0.28}, step=3)
        
        # Check training history has all values
        assert len(trainer._training_history["accuracy"]) == 3
        assert len(trainer._training_history["loss"]) == 3
        
        # Check best metrics are updated correctly
        assert trainer._best_metrics["accuracy"] == 0.85  # Highest accuracy
        assert trainer._best_metrics["loss"] == 0.25      # Lowest loss
    
    def test_should_stop_early_disabled(self, mock_model, basic_training_config):
        """Test early stopping when disabled."""
        basic_training_config.early_stopping = False
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Should never stop early when disabled
        assert not trainer._should_stop_early(0.5, patience_counter=10)
        assert not trainer._should_stop_early(0.1, patience_counter=100)
    
    def test_should_stop_early_enabled(self, mock_model, basic_training_config):
        """Test early stopping when enabled."""
        basic_training_config.early_stopping = True
        basic_training_config.early_stopping_patience = 3
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Should not stop when patience not exceeded
        assert not trainer._should_stop_early(0.5, patience_counter=2)
        
        # Should stop when patience exceeded
        assert trainer._should_stop_early(0.5, patience_counter=3)
        assert trainer._should_stop_early(0.5, patience_counter=5)
    
    @patch('detectors.training.trainers.base_trainer.StandardEvaluator')
    def test_evaluate_uses_centralized_evaluator(self, mock_evaluator_class, mock_model, basic_training_config, sample_dataset):
        """Test that evaluate method uses centralized evaluator."""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_evaluation_result = Mock()
        mock_evaluation_result.conclusion = Mock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_evaluator_class.return_value = mock_evaluator
        
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        trainer._evaluator = mock_evaluator
        
        # Call evaluate
        result = trainer.evaluate(sample_dataset)
        
        # Verify evaluator was called
        mock_evaluator.evaluate.assert_called_once_with(
            model=mock_model,
            dataset=sample_dataset
        )
        
        # Verify result is the conclusion
        assert result == mock_evaluation_result.conclusion
    
    @patch('detectors.training.trainers.base_trainer.StandardEvaluator')
    def test_evaluate_comprehensive(self, mock_evaluator_class, mock_model, basic_training_config, sample_dataset):
        """Test evaluate_comprehensive method."""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_evaluation_result = Mock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_evaluator_class.return_value = mock_evaluator
        
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        trainer._evaluator = mock_evaluator
        
        # Call evaluate_comprehensive
        result = trainer.evaluate_comprehensive(sample_dataset, batch_size=64)
        
        # Verify evaluator was called with all parameters
        mock_evaluator.evaluate.assert_called_once_with(
            model=mock_model,
            dataset=sample_dataset,
            batch_size=64
        )
        
        # Verify result is the full EvaluationResult
        assert result == mock_evaluation_result
    
    def test_create_training_result(self, mock_model, basic_training_config):
        """Test creation of TrainingResult object."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        start_time = datetime.now()
        end_time = datetime.now()
        train_conclusion = Mock()
        val_conclusion = Mock()
        test_conclusion = Mock()
        
        result = trainer._create_training_result(
            start_time=start_time,
            end_time=end_time,
            train_conclusion=train_conclusion,
            validation_conclusion=val_conclusion,
            test_conclusion=test_conclusion,
            best_checkpoint_path="/path/to/checkpoint"
        )
        
        assert isinstance(result, TrainingResult)
        assert result.run_id == basic_training_config.run_id
        assert result.trainer_type == "ConcreteTrainer"
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.train_conclusion == train_conclusion
        assert result.validation_conclusion == val_conclusion
        assert result.test_conclusion == test_conclusion
        assert result.best_checkpoint_path == "/path/to/checkpoint"
        assert result.model_weights == mock_model.store_weights.return_value
        assert result.config == basic_training_config
    
    def test_get_metadata(self, mock_model, basic_training_config):
        """Test metadata generation."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Add some training history and best metrics
        trainer._training_history = {"loss": [0.5, 0.3, 0.2]}
        trainer._best_metrics = {"loss": 0.2, "accuracy": 0.85}
        
        metadata = trainer._get_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata["trainer_class"] == "ConcreteTrainer"
        assert metadata["best_metrics"] == trainer._best_metrics
    
    def test_save_checkpoint(self, mock_model, basic_training_config, tmp_path):
        """Test saving training checkpoint."""
        basic_training_config.output_dir = str(tmp_path)
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Add some training state
        trainer._training_history = {"loss": [0.5, 0.3]}
        trainer._best_metrics = {"loss": 0.3}
        
        checkpoint_path = str(tmp_path / "checkpoint")
        metrics = {"accuracy": 0.85, "loss": 0.25}
        
        trainer.save_checkpoint(checkpoint_path, epoch=5, metrics=metrics)
        
        # Verify model checkpoint was saved
        mock_model.save_checkpoint.assert_called_once_with(checkpoint_path)
        
        # Verify trainer state was saved
        state_file = Path(checkpoint_path) / "trainer_state.json"
        assert state_file.exists()
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        assert state["epoch"] == 5
        assert state["metrics"] == metrics
        assert state["training_history"] == trainer._training_history
        assert state["best_metrics"] == trainer._best_metrics
    
    def test_load_checkpoint(self, mock_model, basic_training_config, tmp_path):
        """Test loading training checkpoint."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Create checkpoint directory and state file
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()
        
        state_data = {
            "epoch": 10,
            "metrics": {"accuracy": 0.90},
            "training_history": {"loss": [0.8, 0.6, 0.4]},
            "best_metrics": {"loss": 0.4, "accuracy": 0.90}
        }
        
        state_file = checkpoint_path / "trainer_state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f)
        
        # Load checkpoint
        loaded_state = trainer.load_checkpoint(str(checkpoint_path))
        
        # Verify model checkpoint was loaded
        mock_model.load_checkpoint.assert_called_once_with(str(checkpoint_path))
        
        # Verify trainer state was restored
        assert trainer._training_history == state_data["training_history"]
        assert trainer._best_metrics == state_data["best_metrics"]
        assert loaded_state == state_data
    
    def test_load_checkpoint_missing_state(self, mock_model, basic_training_config, tmp_path):
        """Test loading checkpoint when trainer state file is missing."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()
        # No trainer_state.json file
        
        loaded_state = trainer.load_checkpoint(str(checkpoint_path))
        
        # Should still load model checkpoint
        mock_model.load_checkpoint.assert_called_once_with(str(checkpoint_path))
        
        # Should return empty dict when state file missing
        assert loaded_state == {}
    
    def test_save_model(self, mock_model, basic_training_config, tmp_path):
        """Test saving trained model."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Add some training data
        trainer._training_history = {"loss": [0.5, 0.3]}
        trainer._best_metrics = {"accuracy": 0.85}
        
        model_path = str(tmp_path / "model")
        trainer.save_model(model_path)
        
        # Verify model was saved
        mock_model.save_checkpoint.assert_called_once_with(model_path)
        
        # Verify metadata was saved
        metadata_file = Path(model_path) / "training_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["training_history"] == trainer._training_history
        assert metadata["best_metrics"] == trainer._best_metrics
        assert "config" in metadata
        assert "metadata" in metadata
    
    def test_load_model(self, mock_model, basic_training_config, tmp_path):
        """Test loading trained model."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Create model directory and metadata
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        metadata = {
            "training_history": {"loss": [0.8, 0.5]},
            "best_metrics": {"accuracy": 0.88},
            "config": {},
            "metadata": {}
        }
        
        metadata_file = model_path / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Load model
        trainer.load_model(str(model_path))
        
        # Verify model was loaded
        mock_model.load_checkpoint.assert_called_once_with(str(model_path))
        
        # Verify metadata was restored
        assert trainer._training_history == metadata["training_history"]
        assert trainer._best_metrics == metadata["best_metrics"]
    
    def test_load_model_missing_metadata(self, mock_model, basic_training_config, tmp_path):
        """Test loading model when metadata file is missing."""
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        model_path = tmp_path / "model"
        model_path.mkdir()
        # No training_metadata.json file
        
        trainer.load_model(str(model_path))
        
        # Should still load model
        mock_model.load_checkpoint.assert_called_once_with(str(model_path))
        
        # Training history should remain empty
        assert trainer._training_history == {}
        assert trainer._best_metrics == {}


class TestBaseTrainerIntegration:
    """Integration tests for BaseTrainer."""
    
    def test_training_workflow(self, mock_model, basic_training_config, sample_dataset, tmp_path):
        """Test complete training workflow."""
        basic_training_config.output_dir = str(tmp_path / "training")
        trainer = ConcreteTrainer(mock_model, basic_training_config)
        
        # Mock evaluator to return proper results
        with patch.object(trainer, '_evaluator') as mock_evaluator:
            mock_result = Mock()
            mock_result.conclusion = Mock()
            mock_evaluator.evaluate.return_value = mock_result
            
            # Run training
            result = trainer.train(
                train_dataset=sample_dataset,
                validation_dataset=sample_dataset,
                test_dataset=sample_dataset
            )
        
        # Verify training result
        assert isinstance(result, TrainingResult)
        assert result.validation_conclusion is not None
        assert result.test_conclusion is not None
        
        # Verify evaluator was called for validation and test
        assert mock_evaluator.evaluate.call_count == 2
    
    def test_checkpoint_workflow(self, mock_model, basic_training_config, tmp_path):
        """Test checkpoint save/load workflow."""
        basic_training_config.output_dir = str(tmp_path)
        trainer1 = ConcreteTrainer(mock_model, basic_training_config)
        
        # Simulate some training progress
        trainer1._log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        trainer1._log_metrics({"loss": 0.3, "accuracy": 0.85}, step=2)
        
        # Save checkpoint
        checkpoint_path = str(tmp_path / "checkpoint")
        trainer1.save_checkpoint(checkpoint_path, epoch=2, metrics={"accuracy": 0.85})
        
        # Create new trainer and load checkpoint
        trainer2 = ConcreteTrainer(mock_model, basic_training_config)
        state = trainer2.load_checkpoint(checkpoint_path)
        
        # Verify state was transferred
        assert state["epoch"] == 2
        assert trainer2._training_history == trainer1._training_history
        assert trainer2._best_metrics == trainer1._best_metrics
    
    def test_model_save_load_workflow(self, mock_model, basic_training_config, tmp_path):
        """Test model save/load workflow."""
        trainer1 = ConcreteTrainer(mock_model, basic_training_config)
        
        # Simulate training
        trainer1._log_metrics({"accuracy": 0.9}, step=1)
        
        # Save model
        model_path = str(tmp_path / "model")
        trainer1.save_model(model_path)
        
        # Create new trainer and load model
        trainer2 = ConcreteTrainer(mock_model, basic_training_config)
        trainer2.load_model(model_path)
        
        # Verify metadata was transferred
        assert trainer2._training_history == trainer1._training_history
        assert trainer2._best_metrics == trainer1._best_metrics 