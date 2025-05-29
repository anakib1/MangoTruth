"""Updated RoBERTa training script using the new training architecture.

This script demonstrates how to train a RoBERTa model for plagiarism detection
using the new training pipeline architecture.
"""

import argparse
import logging
from uuid import uuid4
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv

from detectors.data.datasets import Dataset, TextSample
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.training import HuggingFaceTrainer, HuggingFaceTrainingConfig
from detectors.neptune.nexus import NeptuneNexus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_huggingface_dataset(hf_dataset, selection_size: int, human_label: int = 0, ai_label: int = 3) -> Dataset:
    """Convert HuggingFace dataset to our Dataset format.
    
    Args:
        hf_dataset: HuggingFace dataset
        selection_size: Number of samples per class
        human_label: Label value for human samples
        ai_label: Label value for AI samples
        
    Returns:
        Dataset object with balanced samples
    """
    # Filter by length
    hf_dataset = hf_dataset.filter(lambda x: len(x['output']) < 15000)
    
    # Get human and AI samples
    human_data = hf_dataset.filter(lambda x: x['label'] == human_label)
    ai_data = hf_dataset.filter(lambda x: x['label'] == ai_label)
    
    # Balance dataset
    selection_size = min(selection_size, len(human_data), len(ai_data))
    
    human_samples = human_data.shuffle().select(range(selection_size))
    ai_samples = ai_data.shuffle().select(range(selection_size))
    
    # Convert to our TextSample format
    samples = []
    
    # Add human samples
    for item in human_samples:
        samples.append(TextSample(
            prompt=item.get('prompt', ''),
            output=item['output'],
            author_id=item.get('user_id', 'human'),
            label='human'
        ))
    
    # Add AI samples  
    for item in ai_samples:
        samples.append(TextSample(
            prompt=item.get('prompt', ''),
            output=item['output'],
            author_id=item.get('user_id', 'ai'),
            label='ai'
        ))
    
    return Dataset.from_samples(samples)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fine-tune a transformer model using new architecture.')
    parser.add_argument('--dataset_handle', type=str, default='anakib1/mango-truth', 
                       help='The dataset handle to use.')
    parser.add_argument('--dataset_config', type=str, default='xlsum', 
                       help='The dataset configuration.')
    parser.add_argument('--model_handle', type=str, default='FacebookAI/roberta-base',
                       help='The model handle to use.')
    parser.add_argument('--train_size', type=int, default=5000, 
                       help='Number of training examples per class.')
    parser.add_argument('--test_size', type=int, default=1000, 
                       help='Number of test examples per class.')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for both training and evaluation.')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                       help='Learning rate.')
    parser.add_argument('--output_dir', type=str, default='./training_output', 
                       help='Output directory for training artifacts.')
    parser.add_argument('--save_model_path', type=str, default='./models/roberta_detector',
                       help='Path to save the final trained model.')
    
    args = parser.parse_args()
    
    logger.info(f"Starting RoBERTa training with arguments: {args}")
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset_handle} with config {args.dataset_config}")
    hf_data = load_dataset(args.dataset_handle, args.dataset_config)
    
    # Process training data
    logger.info("Processing training data...")
    train_dataset = process_huggingface_dataset(hf_data['train'], args.train_size)
    
    # Process test data  
    logger.info("Processing test data...")
    test_dataset = process_huggingface_dataset(hf_data['test'], args.test_size)
    
    # Split training data for validation
    train_split, val_split = train_dataset.split([0.7, 0.3], seed=42)
    
    logger.info(f"Dataset sizes - Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_dataset)}")
    
    # Create model configuration
    model_config = HuggingFaceConfig(
        model_name=args.model_handle,
        tokenizer_name=args.model_handle,
        num_labels=2,
        max_length=512,
        cache_dir="./model_cache"
    )
    
    # Create trainable model
    logger.info(f"Initializing model {args.model_handle}...")
    model = TrainableHuggingFaceDetector(model_config)
    
    # Update model's label mapping
    model.model.config.id2label = {0: "human", 1: "ai"}
    model.model.config.label2id = {"human": 0, "ai": 1}
    
    # Create run identifier
    run_id = uuid4()
    run_surname = f"roberta-{args.model_handle.split('/')[-1]}-{args.train_size}-{run_id.hex[:8]}"
    
    # Create training configuration
    training_config = HuggingFaceTrainingConfig(
        # Model info
        model_name=args.model_handle,
        num_labels=2,
        max_length=512,
        
        # Training parameters
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # Evaluation and saving
        validation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        
        # Output
        output_dir=f"{args.output_dir}/{run_surname}",
        
        # Efficiency
        fp16=True,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        
        # Early stopping
        early_stopping=True,
        early_stopping_patience=3,
        
        # Metrics
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Neptune reporting
        push_to_hub=False,
        
        # Run ID
        run_id=run_id
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = HuggingFaceTrainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    result = trainer.train(
        train_dataset=train_split,
        validation_dataset=val_split,
        test_dataset=test_dataset
    )
    
    # Print results
    logger.info(f"Training completed in {result.training_duration_seconds:.2f} seconds")
    logger.info(f"Best checkpoint: {result.best_checkpoint_path}")
    
    # Print metrics
    if result.validation_conclusion:
        val_metrics = result.validation_conclusion.train_conclusion.metrics
        logger.info(f"Validation Metrics:")
        logger.info(f"  Accuracy: {val_metrics.accuracy:.4f}")
        logger.info(f"  F1: {val_metrics.f1:.4f}")
        logger.info(f"  AUC: {val_metrics.auc:.4f}")
        logger.info(f"  Precision: {val_metrics.precision:.4f}")
        logger.info(f"  Recall: {val_metrics.recall:.4f}")
    
    if result.test_conclusion:
        test_metrics = result.test_conclusion.train_conclusion.metrics
        logger.info(f"Test Metrics:")
        logger.info(f"  Accuracy: {test_metrics.accuracy:.4f}")
        logger.info(f"  F1: {test_metrics.f1:.4f}")
        logger.info(f"  AUC: {test_metrics.auc:.4f}")
        logger.info(f"  Precision: {test_metrics.precision:.4f}")
        logger.info(f"  Recall: {test_metrics.recall:.4f}")
    
    # Save final model
    logger.info(f"Saving model to {args.save_model_path}")
    trainer.save_model(args.save_model_path)
    
    # Upload to Neptune (if configured)
    try:
        logger.info("Uploading results to Neptune...")
        nexus = NeptuneNexus()
        
        # Use the train_conclusion from the result directly
        conclusion = result.train_conclusion
        
        extra_data = {
            "model_handle": args.model_handle,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "training_duration": result.training_duration_seconds,
            "total_params": model.get_num_parameters(),
            "trainable_params": model.get_num_trainable_parameters(),
        }
        
        nexus.conclude_run(run_id, conclusion=conclusion, extra_data=extra_data)
        logger.info(f"Successfully uploaded run {run_id} to Neptune")
        
    except Exception as e:
        logger.warning(f"Failed to upload to Neptune: {e}")
    
    # Test loading the saved model from Neptune
    logger.info("Testing model loading from Neptune...")
    
    try:
        # Create a fresh model instance  
        fresh_config = HuggingFaceConfig(
            model_name=args.model_handle,
            tokenizer_name=args.model_handle,
            num_labels=2,
            max_length=512,
            cache_dir="./model_cache"
        )
        
        # Create new model instance
        loaded_model = TrainableHuggingFaceDetector(fresh_config)
        
        # Load weights from Neptune run
        nexus = NeptuneNexus()
        weights_bytes = nexus.load_run_weights(run_id)
        loaded_model.load_weights(weights_bytes)
        
        logger.info(f"Successfully loaded model weights from Neptune run {run_id}")
        
        # Test prediction
        test_text = "Artificial intelligence is transforming how we interact with technology."
        probs = loaded_model.predict_proba(test_text)
        labels = loaded_model.get_labels()
        
        logger.info(f"Test prediction for: '{test_text}'")
        for label, prob in zip(labels, probs):
            logger.info(f"  {label}: {prob:.4f}")
            
    except Exception as e:
        logger.warning(f"Failed to load model from Neptune: {e}")
        logger.info("Using current model for test prediction instead...")
        
        # Fallback to current model
        test_text = "Artificial intelligence is transforming how we interact with technology."
        probs = model.predict_proba(test_text)
        labels = model.get_labels()
        
        logger.info(f"Test prediction for: '{test_text}'")
        for label, prob in zip(labels, probs):
            logger.info(f"  {label}: {prob:.4f}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    load_dotenv()
    main() 