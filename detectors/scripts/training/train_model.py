"""Universal training script using the new training architecture.

This script demonstrates best practices for training models using
the centralized factories and preprocessing utilities.
"""

import argparse
import logging
from pathlib import Path
from uuid import uuid4
from typing import Optional

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.training.configs import HuggingFaceTrainingConfig
from detectors.training.factories import TrainerFactory
from detectors.training.preprocessing import DatasetProcessorFactory
from detectors.neptune.nexus import NeptuneNexus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model_from_config(model_config: dict) -> TrainableHuggingFaceDetector:
    """Create model from configuration.
    
    Args:
        model_config: Model configuration dictionary.
        
    Returns:
        Configured model instance.
    """
    config = HuggingFaceConfig(**model_config)
    model = TrainableHuggingFaceDetector(config)
    
    # Update model's label mapping
    model.model.config.id2label = {0: "human", 1: "ai"}
    model.model.config.label2id = {"human": 0, "ai": 1}
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Universal model training script.')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file.')
    parser.add_argument('--model_config', type=str,
                       help='Path to model configuration file.')
    
    # Data configuration (can override config file)
    parser.add_argument('--dataset_handle', type=str,
                       help='HuggingFace dataset handle.')
    parser.add_argument('--dataset_config', type=str,
                       help='Dataset configuration name.')
    parser.add_argument('--train_size', type=int,
                       help='Number of training examples per class.')
    parser.add_argument('--test_size', type=int,
                       help='Number of test examples per class.')
    
    # Training overrides
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int,
                       help='Training batch size.')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate.')
    
    # Output
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for training artifacts.')
    parser.add_argument('--save_model_path', type=str,
                       help='Path to save the final model.')
    
    # Experiment tracking
    parser.add_argument('--run_name', type=str,
                       help='Name for this training run.')
    parser.add_argument('--no_neptune', action='store_true',
                       help='Disable Neptune logging.')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_data = load_config_from_file(args.config)
    
    # Create training configuration
    training_config_data = config_data.get('training', {})
    
    # Apply command line overrides
    overrides = {
        'dataset_handle': args.dataset_handle,
        'dataset_config': args.dataset_config,
        'train_size': args.train_size,
        'test_size': args.test_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
    }
    
    for key, value in overrides.items():
        if value is not None:
            training_config_data[key] = value
    
    # Set run ID and run name
    run_id = uuid4()
    if args.run_name:
        run_surname = args.run_name
    else:
        model_name = training_config_data.get('model_name', 'model')
        train_size = training_config_data.get('train_size', 'unknown')
        run_surname = f"{model_name.split('/')[-1]}-{train_size}-{run_id.hex[:8]}"
    
    training_config_data['run_id'] = run_id
    if 'output_dir' not in training_config_data:
        training_config_data['output_dir'] = f"./training_output/{run_surname}"
    
    # Create training configuration
    training_config = HuggingFaceTrainingConfig(**training_config_data)
    
    logger.info(f"Training configuration: {training_config.model_name}")
    logger.info(f"Output directory: {training_config.output_dir}")
    
    # Load model configuration
    if args.model_config:
        model_config_data = load_config_from_file(args.model_config)
    else:
        model_config_data = config_data.get('model', {})
        # Use training config model name if not specified
        if 'model_name' not in model_config_data:
            model_config_data['model_name'] = training_config.model_name
            model_config_data['tokenizer_name'] = training_config.model_name
    
    # Create model
    logger.info(f"Creating model: {model_config_data.get('model_name')}")
    model = create_model_from_config(model_config_data)
    
    # Load and process dataset
    dataset_config_data = config_data.get('dataset', {})
    dataset_handle = training_config_data.get('dataset_handle', 'anakib1/mango-truth')
    dataset_config_name = training_config_data.get('dataset_config', 'xlsum')
    
    logger.info(f"Loading dataset {dataset_handle}:{dataset_config_name}")
    hf_data = load_dataset(dataset_handle, dataset_config_name)
    
    # Create dataset processor
    processor_config = dataset_config_data.get('processor', {})
    processor = DatasetProcessorFactory.create_processor('huggingface', **processor_config)
    
    # Process datasets
    train_size = training_config_data.get('train_size', 5000)
    test_size = training_config_data.get('test_size', 1000)
    
    logger.info("Processing training dataset...")
    train_dataset, val_dataset = processor.process_split(
        hf_data['train'], 
        train_size,
        validation_split=0.3,
        seed=42
    )
    
    logger.info("Processing test dataset...")
    test_dataset = processor.process(hf_data['test'], test_size, seed=42)
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = TrainerFactory.create_from_config_type(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    result = trainer.train(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # Print results
    logger.info(f"Training completed in {result.training_duration_seconds:.2f} seconds")
    logger.info(f"Best checkpoint: {result.best_checkpoint_path}")
    
    # Print metrics
    if result.validation_conclusion:
        val_metrics = result.validation_conclusion.train_conclusion.metrics
        logger.info("Validation Metrics:")
        logger.info(f"  Accuracy: {val_metrics.accuracy:.4f}")
        logger.info(f"  F1: {val_metrics.f1:.4f}")
        logger.info(f"  AUC: {val_metrics.auc:.4f}")
    
    if result.test_conclusion:
        test_metrics = result.test_conclusion.train_conclusion.metrics
        logger.info("Test Metrics:")
        logger.info(f"  Accuracy: {test_metrics.accuracy:.4f}")
        logger.info(f"  F1: {test_metrics.f1:.4f}")
        logger.info(f"  AUC: {test_metrics.auc:.4f}")
    
    # Save model
    save_path = args.save_model_path or f"./models/{run_surname}"
    logger.info(f"Saving model to {save_path}")
    trainer.save_model(save_path)
    
    # Upload to Neptune (if enabled)
    if not args.no_neptune:
        try:
            logger.info("Uploading results to Neptune...")
            nexus = NeptuneNexus()
            
            extra_data = {
                "run_surname": run_surname,
                "config_file": args.config,
                "total_params": model.get_num_parameters(),
                "trainable_params": model.get_num_trainable_parameters(),
            }
            
            nexus.conclude_run(run_id, conclusion=result.train_conclusion, extra_data=extra_data)
            logger.info(f"Successfully uploaded run {run_id} to Neptune")
            
        except Exception as e:
            logger.warning(f"Failed to upload to Neptune: {e}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    load_dotenv()
    main() 