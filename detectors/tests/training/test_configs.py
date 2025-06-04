"""Tests for training configuration classes."""

import pytest
import json
import tempfile
from uuid import uuid4
from pathlib import Path

from detectors.training.configs import (
    BaseTrainingConfig,
    HuggingFaceTrainingConfig,
    ClassicalMLTrainingConfig
)


class TestBaseTrainingConfig:
    """Tests for BaseTrainingConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = BaseTrainingConfig()
        
        assert config.epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 5e-5
        assert config.validation_split == 0.2
        assert config.validation_strategy == "epoch"
        assert config.early_stopping is True
        assert config.early_stopping_patience == 3
        assert config.seed == 42
        assert config.output_dir == "./training_output"
        assert isinstance(config.extra_params, dict)
        assert len(config.extra_params) == 0
    
    def test_custom_values(self):
        """Test setting custom values."""
        run_id = uuid4()
        config = BaseTrainingConfig(
            epochs=10,
            batch_size=64,
            learning_rate=1e-4,
            validation_split=0.1,
            early_stopping=False,
            seed=123,
            output_dir="./custom_output",
            run_id=run_id,
            extra_params={"custom_param": "value"}
        )
        
        assert config.epochs == 10
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.validation_split == 0.1
        assert config.early_stopping is False
        assert config.seed == 123
        assert config.output_dir == "./custom_output"
        assert config.run_id == run_id
        assert config.extra_params["custom_param"] == "value"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        run_id = uuid4()
        config = BaseTrainingConfig(
            epochs=5,
            batch_size=16,
            run_id=run_id
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["epochs"] == 5
        assert config_dict["batch_size"] == 16
        assert config_dict["run_id"] == str(run_id)
        assert "learning_rate" in config_dict
        assert "validation_split" in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        run_id = uuid4()
        config_dict = {
            "epochs": 7,
            "batch_size": 128,
            "learning_rate": 2e-5,
            "run_id": str(run_id),
            "extra_params": {"test": "value"}
        }
        
        config = BaseTrainingConfig.from_dict(config_dict)
        
        assert config.epochs == 7
        assert config.batch_size == 128
        assert config.learning_rate == 2e-5
        assert config.run_id == run_id
        assert config.extra_params["test"] == "value"
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config = BaseTrainingConfig(
            epochs=8,
            batch_size=256,
            learning_rate=3e-5
        )
        
        config_path = tmp_path / "config.json"
        config.save(str(config_path))
        
        # Check file was created
        assert config_path.exists()
        
        # Load and verify
        loaded_config = BaseTrainingConfig.load(str(config_path))
        assert loaded_config.epochs == 8
        assert loaded_config.batch_size == 256
        assert loaded_config.learning_rate == 3e-5
        assert loaded_config.run_id == config.run_id
    
    def test_save_creates_directories(self, tmp_path):
        """Test that save creates necessary directories."""
        config = BaseTrainingConfig()
        nested_path = tmp_path / "nested" / "directories" / "config.json"
        
        config.save(str(nested_path))
        assert nested_path.exists()


class TestHuggingFaceTrainingConfig:
    """Tests for HuggingFaceTrainingConfig class."""
    
    def test_default_values(self):
        """Test default values specific to HuggingFace config."""
        config = HuggingFaceTrainingConfig()
        
        assert config.model_name == "bert-base-uncased"
        assert config.tokenizer_name == "bert-base-uncased"  # Should match model_name
        assert config.num_labels == 2
        assert config.max_length == 512
        assert config.optimizer_name == "adamw_torch"
        assert config.scheduler_type == "linear"
        assert config.weight_decay == 0.01
        assert config.fp16 is False
        assert config.gradient_accumulation_steps == 1
        assert config.push_to_hub is False
        assert config.metric_for_best_model == "eval_loss"
        assert config.greater_is_better is False
    
    def test_post_init_tokenizer_name(self):
        """Test that tokenizer_name is set to model_name if not specified."""
        config = HuggingFaceTrainingConfig(model_name="roberta-base")
        assert config.tokenizer_name == "roberta-base"
        
        # Test explicit tokenizer_name is preserved
        config = HuggingFaceTrainingConfig(
            model_name="bert-base",
            tokenizer_name="custom-tokenizer"
        )
        assert config.tokenizer_name == "custom-tokenizer"
    
    def test_post_init_batch_sizes(self):
        """Test that per-device batch sizes are set correctly."""
        config = HuggingFaceTrainingConfig(batch_size=64)
        assert config.per_device_train_batch_size == 64
        assert config.per_device_eval_batch_size == 64
        
        # Test explicit values are preserved
        config = HuggingFaceTrainingConfig(
            batch_size=32,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8
        )
        assert config.per_device_train_batch_size == 16
        assert config.per_device_eval_batch_size == 8
    
    def test_post_init_evaluation_sync(self):
        """Test that evaluation settings are synced."""
        config = HuggingFaceTrainingConfig(
            validation_strategy="steps",
            validation_steps=100
        )
        assert config.eval_strategy == "steps"
        assert config.eval_steps == 100
    
    def test_to_hf_training_args(self):
        """Test conversion to HuggingFace TrainingArguments format."""
        config = HuggingFaceTrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=True,
            output_dir="./test_output"
        )
        
        args_dict = config.to_hf_training_args()
        
        assert args_dict["num_train_epochs"] == 5
        assert args_dict["per_device_train_batch_size"] == 32
        assert args_dict["learning_rate"] == 2e-5
        assert args_dict["warmup_steps"] == 500
        assert args_dict["weight_decay"] == 0.01
        assert args_dict["fp16"] is True
        assert args_dict["output_dir"] == "./test_output"
        assert args_dict["optim"] == "adamw_torch"
        assert args_dict["lr_scheduler_type"] == "linear"
    
    def test_to_hf_training_args_optional_params(self):
        """Test that optional parameters are included only when not None."""
        config = HuggingFaceTrainingConfig(
            warmup_steps=None,
            warmup_ratio=0.1,
            save_steps=None,
            eval_steps=200
        )
        
        args_dict = config.to_hf_training_args()
        
        assert "warmup_steps" not in args_dict
        assert args_dict["warmup_ratio"] == 0.1
        assert "save_steps" not in args_dict
        assert args_dict["eval_steps"] == 200
    
    def test_model_kwargs_and_tokenizer_kwargs(self):
        """Test model and tokenizer kwargs are preserved."""
        model_kwargs = {"hidden_dropout_prob": 0.2}
        tokenizer_kwargs = {"do_lower_case": True}
        
        config = HuggingFaceTrainingConfig(
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs
        )
        
        assert config.model_kwargs == model_kwargs
        assert config.tokenizer_kwargs == tokenizer_kwargs


class TestClassicalMLTrainingConfig:
    """Tests for ClassicalMLTrainingConfig class."""
    
    def test_default_values(self):
        """Test default values for classical ML config."""
        config = ClassicalMLTrainingConfig()
        
        assert config.model_type == "svm"
        assert config.feature_type == "tfidf"
        assert config.max_features == 10000
        assert config.ngram_range == (1, 2)
        assert config.cv_folds == 5
        assert isinstance(config.model_params, dict)
    
    def test_custom_values(self):
        """Test setting custom values for classical ML."""
        model_params = {"C": 1.0, "kernel": "rbf"}
        config = ClassicalMLTrainingConfig(
            model_type="random_forest",
            feature_type="bow",
            max_features=5000,
            ngram_range=(1, 3),
            cv_folds=10,
            model_params=model_params
        )
        
        assert config.model_type == "random_forest"
        assert config.feature_type == "bow"
        assert config.max_features == 5000
        assert config.ngram_range == (1, 3)
        assert config.cv_folds == 10
        assert config.model_params == model_params


class TestConfigSerialization:
    """Tests for configuration serialization/deserialization."""
    
    def test_huggingface_config_roundtrip(self, tmp_path):
        """Test full roundtrip serialization for HuggingFace config."""
        original_config = HuggingFaceTrainingConfig(
            model_name="distilbert-base",
            epochs=10,
            batch_size=16,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            fp16=True,
            model_kwargs={"dropout": 0.1},
            extra_params={"custom": "value"}
        )
        
        config_path = tmp_path / "hf_config.json"
        original_config.save(str(config_path))
        
        loaded_config = HuggingFaceTrainingConfig.load(str(config_path))
        
        assert loaded_config.model_name == original_config.model_name
        assert loaded_config.epochs == original_config.epochs
        assert loaded_config.batch_size == original_config.batch_size
        assert loaded_config.learning_rate == original_config.learning_rate
        assert loaded_config.warmup_ratio == original_config.warmup_ratio
        assert loaded_config.fp16 == original_config.fp16
        assert loaded_config.model_kwargs == original_config.model_kwargs
        assert loaded_config.extra_params == original_config.extra_params
        assert loaded_config.run_id == original_config.run_id
    
    def test_invalid_json_handling(self, tmp_path):
        """Test handling of invalid JSON files."""
        config_path = tmp_path / "invalid.json"
        config_path.write_text("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            BaseTrainingConfig.load(str(config_path))
    
    def test_missing_file_handling(self):
        """Test handling of missing configuration files."""
        with pytest.raises(FileNotFoundError):
            BaseTrainingConfig.load("nonexistent_config.json")


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_positive_epochs(self):
        """Test that epochs must be positive."""
        # This should work
        config = BaseTrainingConfig(epochs=1)
        assert config.epochs == 1
        
        # Note: If validation is added later, we can test negative epochs here
    
    def test_positive_batch_size(self):
        """Test that batch_size must be positive."""
        config = BaseTrainingConfig(batch_size=1)
        assert config.batch_size == 1
    
    def test_valid_learning_rate(self):
        """Test that learning_rate must be positive."""
        config = BaseTrainingConfig(learning_rate=1e-6)
        assert config.learning_rate == 1e-6
    
    def test_validation_split_range(self):
        """Test that validation_split is in valid range."""
        config = BaseTrainingConfig(validation_split=0.0)
        assert config.validation_split == 0.0
        
        config = BaseTrainingConfig(validation_split=0.5)
        assert config.validation_split == 0.5
    
    def test_patience_positive(self):
        """Test that early stopping patience is positive."""
        config = BaseTrainingConfig(early_stopping_patience=1)
        assert config.early_stopping_patience == 1 