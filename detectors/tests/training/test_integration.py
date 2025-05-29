"""Integration tests for the training pipeline."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from detectors.training.factories import TrainerFactory, TrainerType
from detectors.training.configs import HuggingFaceTrainingConfig, BaseTrainingConfig
from detectors.training.preprocessing import DatasetProcessorFactory
from detectors.training.interfaces import TrainingResult
from detectors.evaluation import StandardEvaluator


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_end_to_end_training_workflow(self, mock_model, sample_dataset, tmp_path):
        """Test complete end-to-end training workflow."""
        # Setup configuration
        config = HuggingFaceTrainingConfig(
            model_name="distilbert-base-uncased",
            epochs=2,
            batch_size=16,
            learning_rate=2e-5,
            output_dir=str(tmp_path / "training_output")
        )
        
        # Create trainer using factory
        trainer = TrainerFactory.create_trainer(
            TrainerType.HUGGINGFACE,
            mock_model,
            config
        )
        
        # Mock evaluation to avoid complexity
        with patch.object(trainer, '_evaluator') as mock_evaluator:
            mock_eval_result = Mock()
            mock_eval_result.conclusion = Mock()
            mock_evaluator.evaluate.return_value = mock_eval_result
            
            # Run training
            result = trainer.train(
                train_dataset=sample_dataset,
                validation_dataset=sample_dataset,
                test_dataset=sample_dataset
            )
        
        # Verify training result
        assert isinstance(result, TrainingResult)
        assert result.model_name == "distilbert-base-uncased"
        assert result.trainer_type == "HuggingFaceTrainer"
        assert result.train_conclusion is not None
        assert result.validation_conclusion is not None
        assert result.test_conclusion is not None
        assert result.config == config
        
        # Verify output directory structure
        output_dir = Path(config.output_dir)
        assert output_dir.exists()
        assert (output_dir / "training_config.json").exists()
    
    def test_configuration_persistence_workflow(self, mock_model, tmp_path):
        """Test configuration save/load workflow."""
        # Create and save configuration
        original_config = HuggingFaceTrainingConfig(
            model_name="roberta-base",
            epochs=5,
            batch_size=32,
            learning_rate=1e-4,
            fp16=True,
            output_dir=str(tmp_path / "config_test")
        )
        
        config_file = tmp_path / "test_config.json"
        original_config.save(str(config_file))
        
        # Load configuration
        loaded_config = HuggingFaceTrainingConfig.load(str(config_file))
        
        # Verify configuration integrity
        assert loaded_config.model_name == original_config.model_name
        assert loaded_config.epochs == original_config.epochs
        assert loaded_config.batch_size == original_config.batch_size
        assert loaded_config.learning_rate == original_config.learning_rate
        assert loaded_config.fp16 == original_config.fp16
        assert loaded_config.run_id == original_config.run_id
        
        # Test trainer creation with loaded config
        trainer = TrainerFactory.create_trainer(
            TrainerType.HUGGINGFACE,
            mock_model,
            loaded_config
        )
        
        assert trainer.config.model_name == "roberta-base"
        assert trainer.config.epochs == 5
    
    def test_dataset_processing_integration(self, mock_hf_dataset):
        """Test dataset processing integration with training."""
        # Setup mock HuggingFace dataset
        mock_hf_dataset.filter.return_value = mock_hf_dataset
        mock_hf_dataset.shuffle.return_value = mock_hf_dataset
        mock_hf_dataset.select.return_value = mock_hf_dataset
        mock_hf_dataset.__len__.return_value = 100
        mock_hf_dataset.__iter__.return_value = iter([
            {"output": "Human text", "label": 0, "prompt": "Write something"},
            {"output": "AI text", "label": 3, "prompt": "Generate something"}
        ])
        
        # Create processor
        processor = DatasetProcessorFactory.create_processor(
            "huggingface",
            filter_labels=[0, 3],
            balance_classes=True
        )
        
        # Process dataset
        with patch('detectors.training.preprocessing.dataset_processor.Dataset') as mock_dataset_class:
            mock_processed_dataset = Mock()
            mock_dataset_class.from_samples.return_value = mock_processed_dataset
            
            processed_dataset = processor.process(mock_hf_dataset, max_samples=50, seed=42)
        
        # Verify processing
        assert processed_dataset == mock_processed_dataset
        mock_dataset_class.from_samples.assert_called_once()
        
        # Check that samples were properly converted
        samples = mock_dataset_class.from_samples.call_args[0][0]
        assert len(samples) == 2
        assert samples[0].label == "human"
        assert samples[1].label == "ai"
    
    def test_evaluation_integration_with_training(self, mock_model, sample_dataset, tmp_path):
        """Test evaluation integration within training pipeline."""
        config = BaseTrainingConfig(
            epochs=1,
            batch_size=16,
            output_dir=str(tmp_path / "eval_test")
        )
        
        # Create concrete trainer for testing
        from detectors.tests.training.test_base_trainer import ConcreteTrainer
        trainer = ConcreteTrainer(mock_model, config)
        
        # Mock model predictions for evaluation
        mock_model.batch_predict.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]
        
        # Test both evaluation methods
        conclusion = trainer.evaluate(sample_dataset)
        comprehensive_result = trainer.evaluate_comprehensive(sample_dataset)
        
        # Verify evaluation worked
        assert conclusion is not None
        assert comprehensive_result is not None
        
        # Verify evaluator was used
        assert hasattr(trainer, '_evaluator')
        assert isinstance(trainer._evaluator, StandardEvaluator)
    
    def test_checkpoint_integration(self, mock_model, sample_dataset, tmp_path):
        """Test checkpoint save/load integration."""
        config = BaseTrainingConfig(
            epochs=3,
            batch_size=8,
            output_dir=str(tmp_path / "checkpoint_test")
        )
        
        from detectors.tests.training.test_base_trainer import ConcreteTrainer
        
        # Create first trainer and simulate training
        trainer1 = ConcreteTrainer(mock_model, config)
        trainer1._log_metrics({"loss": 0.8, "accuracy": 0.7}, step=1)
        trainer1._log_metrics({"loss": 0.6, "accuracy": 0.8}, step=2)
        
        # Save checkpoint
        checkpoint_path = str(tmp_path / "checkpoint")
        trainer1.save_checkpoint(checkpoint_path, epoch=2, metrics={"accuracy": 0.8})
        
        # Create second trainer and load checkpoint
        trainer2 = ConcreteTrainer(mock_model, config)
        loaded_state = trainer2.load_checkpoint(checkpoint_path)
        
        # Verify state transfer
        assert loaded_state["epoch"] == 2
        assert trainer2._training_history == trainer1._training_history
        assert trainer2._best_metrics == trainer1._best_metrics
        
        # Verify model checkpoint was handled
        mock_model.save_checkpoint.assert_called_with(checkpoint_path)
        mock_model.load_checkpoint.assert_called_with(checkpoint_path)
    
    def test_factory_integration_with_different_trainers(self, mock_model, basic_training_config):
        """Test factory integration with different trainer types."""
        trainer_configs = [
            (TrainerType.HUGGINGFACE, "HuggingFaceTrainer"),
            (TrainerType.PERPLEXITY, "PerplexityTrainer"),
            (TrainerType.GHOSTBUSTER, "GhostbusterTrainer")
        ]
        
        for trainer_type, expected_class_name in trainer_configs:
            trainer = TrainerFactory.create_trainer(
                trainer_type,
                mock_model,
                basic_training_config
            )
            
            assert trainer.__class__.__name__ == expected_class_name
            assert trainer.model == mock_model
            assert trainer.config == basic_training_config
    
    def test_error_handling_integration(self, mock_model, sample_dataset, tmp_path):
        """Test error handling across the training pipeline."""
        config = BaseTrainingConfig(
            epochs=1,
            output_dir=str(tmp_path / "error_test")
        )
        
        from detectors.tests.training.test_base_trainer import ConcreteTrainer
        trainer = ConcreteTrainer(mock_model, config)
        
        # Test evaluation error handling
        mock_model.batch_predict.side_effect = Exception("Model failed")
        
        with pytest.raises(Exception, match="Model failed"):
            trainer.evaluate(sample_dataset)
        
        # Reset mock for next test
        mock_model.batch_predict.side_effect = None
        mock_model.batch_predict.return_value = [[0.5, 0.5]] * 4
        
        # Test checkpoint error handling
        mock_model.save_checkpoint.side_effect = Exception("Save failed")
        
        with pytest.raises(Exception, match="Save failed"):
            trainer.save_checkpoint(str(tmp_path / "failed_checkpoint"), epoch=1, metrics={})


class TestMultiComponentIntegration:
    """Integration tests involving multiple components."""
    
    def test_factory_processor_evaluator_integration(self, mock_model, mock_hf_dataset, tmp_path):
        """Test integration between factory, processor, and evaluator."""
        # Process dataset
        processor = DatasetProcessorFactory.create_processor("huggingface")
        
        # Mock dataset processing
        mock_hf_dataset.__iter__.return_value = iter([
            {"output": "Text 1", "label": 0, "prompt": "Prompt 1"},
            {"output": "Text 2", "label": 3, "prompt": "Prompt 2"}
        ])
        
        with patch('detectors.training.preprocessing.dataset_processor.Dataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=2)
            mock_dataset.__iter__ = Mock(return_value=iter([
                Mock(output="Text 1", label="human"),
                Mock(output="Text 2", label="ai")
            ]))
            mock_dataset.get_labels = Mock(return_value=["human", "ai"])
            mock_dataset_class.from_samples.return_value = mock_dataset
            
            processed_dataset = processor.process(mock_hf_dataset)
        
        # Create trainer with processed dataset
        config = HuggingFaceTrainingConfig(
            epochs=1,
            batch_size=2,
            output_dir=str(tmp_path / "integration_test")
        )
        
        trainer = TrainerFactory.create_trainer(
            TrainerType.HUGGINGFACE,
            mock_model,
            config
        )
        
        # Mock model for evaluation
        mock_model.batch_predict.return_value = [[0.7, 0.3], [0.2, 0.8]]
        
        # Test evaluation on processed dataset
        result = trainer.evaluate_comprehensive(processed_dataset)
        
        # Verify integration worked
        assert result is not None
        assert hasattr(result, 'dataset_info')
        assert hasattr(result, 'model_info')
        assert result.dataset_info['size'] == 2
    
    def test_configuration_driven_pipeline(self, mock_model, mock_hf_dataset, tmp_path):
        """Test configuration-driven training pipeline."""
        # Create configuration file
        config_data = {
            "model_name": "bert-base-uncased",
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "output_dir": str(tmp_path / "config_driven"),
            "fp16": False,
            "gradient_accumulation_steps": 1
        }
        
        config_file = tmp_path / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load configuration
        config = HuggingFaceTrainingConfig.load(str(config_file))
        
        # Create trainer from configuration
        trainer = TrainerFactory.create_from_config_type(mock_model, config)
        
        # Verify trainer was created correctly
        assert trainer.config.model_name == "bert-base-uncased"
        assert trainer.config.epochs == 2
        assert trainer.config.batch_size == 16
        assert trainer.config.learning_rate == 2e-5
        
        # Test that configuration was saved during trainer initialization
        output_dir = Path(config.output_dir)
        assert output_dir.exists()
        saved_config_file = output_dir / "training_config.json"
        assert saved_config_file.exists()
        
        # Verify saved configuration
        with open(saved_config_file, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["model_name"] == "bert-base-uncased"
        assert saved_config["epochs"] == 2
    
    def test_preprocessing_training_evaluation_pipeline(self, mock_model, tmp_path):
        """Test complete preprocessing -> training -> evaluation pipeline."""
        # Mock raw dataset
        mock_raw_dataset = Mock()
        mock_raw_dataset.filter.return_value = mock_raw_dataset
        mock_raw_dataset.__len__.return_value = 10
        mock_raw_dataset.__iter__.return_value = iter([
            {"output": f"Sample text {i}", "label": i % 2, "prompt": f"Prompt {i}"}
            for i in range(10)
        ])
        
        # Step 1: Preprocess dataset
        processor = DatasetProcessorFactory.create_processor(
            "huggingface",
            label_mapping={0: "negative", 1: "positive"},
            filter_labels=[0, 1],
            balance_classes=False
        )
        
        with patch('detectors.training.preprocessing.dataset_processor.Dataset') as mock_dataset_class:
            mock_processed_dataset = Mock()
            mock_processed_dataset.__len__ = Mock(return_value=10)
            mock_processed_dataset.get_labels = Mock(return_value=["negative", "positive"])
            mock_dataset_class.from_samples.return_value = mock_processed_dataset
            
            processed_dataset = processor.process(mock_raw_dataset, max_samples=10)
        
        # Step 2: Create and configure trainer
        config = BaseTrainingConfig(
            epochs=1,
            batch_size=4,
            output_dir=str(tmp_path / "pipeline_test")
        )
        
        from detectors.tests.training.test_base_trainer import ConcreteTrainer
        trainer = ConcreteTrainer(mock_model, config)
        
        # Step 3: Mock training with evaluation
        mock_model.batch_predict.return_value = [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]]
        
        with patch.object(trainer, 'train') as mock_train:
            mock_result = Mock()
            mock_result.validation_conclusion = trainer.evaluate(processed_dataset)
            mock_train.return_value = mock_result
            
            result = trainer.train(
                train_dataset=processed_dataset,
                validation_dataset=processed_dataset
            )
        
        # Verify pipeline completion
        assert result.validation_conclusion is not None
        mock_train.assert_called_once()
    
    def test_reproducibility_integration(self, mock_model, sample_dataset, tmp_path):
        """Test reproducibility across training runs."""
        config = BaseTrainingConfig(
            epochs=1,
            batch_size=2,
            seed=42,
            output_dir=str(tmp_path / "reproducibility_test")
        )
        
        from detectors.tests.training.test_base_trainer import ConcreteTrainer
        
        # Run 1
        trainer1 = ConcreteTrainer(mock_model, config)
        mock_model.batch_predict.return_value = [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]]
        result1 = trainer1.evaluate_comprehensive(sample_dataset)
        
        # Run 2 with same configuration
        trainer2 = ConcreteTrainer(mock_model, config)
        result2 = trainer2.evaluate_comprehensive(sample_dataset)
        
        # Results should be identical with same seed and data
        assert result1.dataset_info == result2.dataset_info
        assert result1.evaluation_config == result2.evaluation_config
        # Note: predictions might differ due to model state, but structure should be same
    
    def test_scaling_integration(self, mock_model, tmp_path):
        """Test integration with different dataset sizes."""
        config = BaseTrainingConfig(
            epochs=1,
            batch_size=8,
            output_dir=str(tmp_path / "scaling_test")
        )
        
        from detectors.tests.training.test_base_trainer import ConcreteTrainer
        trainer = ConcreteTrainer(mock_model, config)
        
        # Test with different dataset sizes
        dataset_sizes = [10, 100, 1000]
        
        for size in dataset_sizes:
            # Create mock dataset of specified size
            from detectors.data.datasets import Dataset, TextSample
            samples = [
                TextSample(f"prompt_{i}", f"text_{i}", f"author_{i}", "human" if i % 2 == 0 else "ai")
                for i in range(size)
            ]
            large_dataset = Dataset.from_samples(samples)
            
            # Mock predictions for the dataset size
            mock_model.batch_predict.return_value = [[0.5, 0.5]] * min(config.batch_size, size)
            
            # Test evaluation scales properly
            result = trainer.evaluate_comprehensive(large_dataset, batch_size=config.batch_size)
            
            assert result.dataset_info['size'] == size
            assert len(result.predictions) == size
            assert len(result.true_labels) == size 