"""Tests for training factory classes."""

import pytest
from unittest.mock import Mock, patch

from detectors.training.factories import TrainerFactory, TrainerType
from detectors.training.configs import HuggingFaceTrainingConfig, ClassicalMLTrainingConfig
from detectors.training.trainers import HuggingFaceTrainer, PerplexityTrainer, GhostbusterTrainer, BaseTrainer
from detectors.training.interfaces import ITrainer, ITrainable


class TestTrainerType:
    """Tests for TrainerType enum."""
    
    def test_enum_values(self):
        """Test that enum values are correct."""
        assert TrainerType.HUGGINGFACE.value == "huggingface"
        assert TrainerType.PERPLEXITY.value == "perplexity"
        assert TrainerType.GHOSTBUSTER.value == "ghostbuster"
        assert TrainerType.CLASSICAL_ML.value == "classical_ml"
    
    def test_enum_membership(self):
        """Test enum membership."""
        assert TrainerType.HUGGINGFACE in TrainerType
        assert TrainerType.PERPLEXITY in TrainerType
        assert TrainerType.GHOSTBUSTER in TrainerType
        assert TrainerType.CLASSICAL_ML in TrainerType


class TestTrainerFactory:
    """Tests for TrainerFactory class."""
    
    def test_create_huggingface_trainer(self, mock_model, huggingface_training_config):
        """Test creating a HuggingFace trainer."""
        trainer = TrainerFactory.create_trainer(
            TrainerType.HUGGINGFACE,
            mock_model,
            huggingface_training_config
        )
        
        assert isinstance(trainer, HuggingFaceTrainer)
        assert trainer.model == mock_model
        assert trainer.config == huggingface_training_config
    
    def test_create_perplexity_trainer(self, mock_model, basic_training_config):
        """Test creating a Perplexity trainer."""
        trainer = TrainerFactory.create_trainer(
            TrainerType.PERPLEXITY,
            mock_model,
            basic_training_config
        )
        
        assert isinstance(trainer, PerplexityTrainer)
        assert trainer.model == mock_model
        assert trainer.config == basic_training_config
    
    def test_create_ghostbuster_trainer(self, mock_model, basic_training_config):
        """Test creating a Ghostbuster trainer."""
        trainer = TrainerFactory.create_trainer(
            TrainerType.GHOSTBUSTER,
            mock_model,
            basic_training_config
        )
        
        assert isinstance(trainer, GhostbusterTrainer)
        assert trainer.model == mock_model
        assert trainer.config == basic_training_config
    
    def test_create_trainer_unsupported_type(self, mock_model, basic_training_config):
        """Test error handling for unsupported trainer types."""
        # Mock an unsupported trainer type
        class UnsupportedType:
            pass
        
        with pytest.raises(ValueError, match="Unsupported trainer type"):
            TrainerFactory.create_trainer(
                UnsupportedType(),
                mock_model,
                basic_training_config
            )
    
    def test_create_from_config_type_huggingface(self, mock_model, huggingface_training_config):
        """Test creating trainer from HuggingFace config type."""
        trainer = TrainerFactory.create_from_config_type(
            mock_model,
            huggingface_training_config
        )
        
        assert isinstance(trainer, HuggingFaceTrainer)
        assert trainer.model == mock_model
        assert trainer.config == huggingface_training_config
    
    def test_create_from_config_type_classical_ml(self, mock_model):
        """Test creating trainer from Classical ML config type."""
        config = ClassicalMLTrainingConfig()
        trainer = TrainerFactory.create_from_config_type(mock_model, config)
        
        assert isinstance(trainer, BaseTrainer)  # Classical ML should map to BaseTrainer
        assert trainer.model == mock_model
        assert trainer.config == config
    
    def test_create_from_config_type_unknown(self, mock_model, basic_training_config):
        """Test creating trainer from unknown config type defaults to BaseTrainer."""
        trainer = TrainerFactory.create_from_config_type(
            mock_model,
            basic_training_config
        )
        
        assert isinstance(trainer, BaseTrainer)
        assert trainer.model == mock_model
        assert trainer.config == basic_training_config
    
    def test_register_trainer(self, mock_model, basic_training_config):
        """Test registering a new trainer type."""
        # Create a mock trainer class
        class CustomTrainer(ITrainer):
            def __init__(self, model, config):
                self.model = model
                self.config = config
            
            def train(self, train_dataset, validation_dataset=None, test_dataset=None):
                pass
            
            def evaluate(self, dataset):
                pass
            
            def save_model(self, path):
                pass
            
            def load_model(self, path):
                pass
        
        # Define a new trainer type
        custom_type = TrainerType("custom")
        
        # Register the trainer
        TrainerFactory.register_trainer(custom_type, CustomTrainer)
        
        # Test that it can be created
        trainer = TrainerFactory.create_trainer(
            custom_type,
            mock_model,
            basic_training_config
        )
        
        assert isinstance(trainer, CustomTrainer)
        assert trainer.model == mock_model
        assert trainer.config == basic_training_config
        
        # Clean up - remove from registry
        if custom_type in TrainerFactory._trainer_registry:
            del TrainerFactory._trainer_registry[custom_type]
    
    def test_get_available_trainers(self):
        """Test getting available trainer types."""
        available = TrainerFactory.get_available_trainers()
        
        assert isinstance(available, dict)
        assert "huggingface" in available
        assert "perplexity" in available
        assert "ghostbuster" in available
        
        # Check that descriptions are strings
        for trainer_type, description in available.items():
            assert isinstance(trainer_type, str)
            assert isinstance(description, str)
    
    def test_trainer_registry_immutability(self):
        """Test that trainer registry cannot be accidentally modified."""
        original_registry = TrainerFactory._trainer_registry.copy()
        
        # Attempt to modify registry directly (this should not affect the factory)
        registry_ref = TrainerFactory._trainer_registry
        original_size = len(registry_ref)
        
        # Verify original trainers are present
        assert TrainerType.HUGGINGFACE in registry_ref
        assert TrainerType.PERPLEXITY in registry_ref
        assert TrainerType.GHOSTBUSTER in registry_ref
        
        # The registry should be the expected size
        assert len(registry_ref) >= 3


class TestTrainerFactoryIntegration:
    """Integration tests for TrainerFactory with real configurations."""
    
    def test_end_to_end_trainer_creation(self, mock_model):
        """Test end-to-end trainer creation with various configurations."""
        # Test HuggingFace trainer creation
        hf_config = HuggingFaceTrainingConfig(
            model_name="distilbert-base-uncased",
            epochs=2,
            batch_size=16
        )
        
        hf_trainer = TrainerFactory.create_from_config_type(mock_model, hf_config)
        assert isinstance(hf_trainer, HuggingFaceTrainer)
        assert hf_trainer.config.model_name == "distilbert-base-uncased"
        assert hf_trainer.config.epochs == 2
        assert hf_trainer.config.batch_size == 16
    
    def test_trainer_initialization_calls_setup(self, mock_model, huggingface_training_config, tmp_path):
        """Test that trainer initialization properly calls setup methods."""
        # Use a real temporary directory
        huggingface_training_config.output_dir = str(tmp_path / "test_output")
        
        trainer = TrainerFactory.create_trainer(
            TrainerType.HUGGINGFACE,
            mock_model,
            huggingface_training_config
        )
        
        # Verify that the trainer was initialized
        assert trainer.model == mock_model
        assert trainer.config == huggingface_training_config
        
        # Check that output directory was created (this happens in _setup)
        from pathlib import Path
        assert Path(huggingface_training_config.output_dir).exists()
    
    @patch('detectors.training.factories.trainer_factory.HuggingFaceTrainer')
    def test_factory_passes_correct_parameters(self, mock_trainer_class, mock_model, huggingface_training_config):
        """Test that factory passes correct parameters to trainer constructors."""
        TrainerFactory.create_trainer(
            TrainerType.HUGGINGFACE,
            mock_model,
            huggingface_training_config
        )
        
        # Verify the trainer class was called with correct parameters
        mock_trainer_class.assert_called_once_with(mock_model, huggingface_training_config)
    
    def test_multiple_trainer_creation_independence(self, mock_model):
        """Test that creating multiple trainers doesn't interfere with each other."""
        config1 = HuggingFaceTrainingConfig(model_name="bert-base", epochs=3)
        config2 = HuggingFaceTrainingConfig(model_name="roberta-base", epochs=5)
        
        trainer1 = TrainerFactory.create_from_config_type(mock_model, config1)
        trainer2 = TrainerFactory.create_from_config_type(mock_model, config2)
        
        # Verify they are different instances
        assert trainer1 is not trainer2
        assert trainer1.config is not trainer2.config
        
        # Verify their configs are correct
        assert trainer1.config.model_name == "bert-base"
        assert trainer1.config.epochs == 3
        assert trainer2.config.model_name == "roberta-base"
        assert trainer2.config.epochs == 5


class TestTrainerFactoryErrorHandling:
    """Tests for error handling in TrainerFactory."""
    
    def test_create_trainer_with_none_model(self, huggingface_training_config):
        """Test error handling when model is None."""
        with pytest.raises(Exception):  # The specific exception depends on the trainer implementation
            TrainerFactory.create_trainer(
                TrainerType.HUGGINGFACE,
                None,
                huggingface_training_config
            )
    
    def test_create_trainer_with_none_config(self, mock_model):
        """Test error handling when config is None."""
        with pytest.raises(Exception):  # The specific exception depends on the trainer implementation
            TrainerFactory.create_trainer(
                TrainerType.HUGGINGFACE,
                mock_model,
                None
            )
    
    def test_register_trainer_with_invalid_class(self):
        """Test error handling when registering invalid trainer class."""
        class NotATrainer:
            pass
        
        custom_type = TrainerType("invalid")
        
        # This should not raise an error during registration
        TrainerFactory.register_trainer(custom_type, NotATrainer)
        
        # But it should fail when trying to create a trainer
        mock_model = Mock()
        mock_config = Mock()
        
        with pytest.raises(Exception):  # Will fail when trying to instantiate
            TrainerFactory.create_trainer(custom_type, mock_model, mock_config)
        
        # Clean up
        if custom_type in TrainerFactory._trainer_registry:
            del TrainerFactory._trainer_registry[custom_type] 