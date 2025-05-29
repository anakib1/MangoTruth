"""Updated Ghostbuster training script using the new training architecture.

This script demonstrates how to train a Ghostbuster model for plagiarism detection
using the new training pipeline architecture.
"""

import argparse
import logging
from uuid import uuid4
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from detectors.data.datasets import Dataset, TextSample
from detectors.models.zoo.trainable_implementations import TrainableGhostbusterDetector
from detectors.training.trainers.ghostbuster_trainer import GhostbusterTrainer, GhostbusterTrainingConfig
from detectors.neptune.nexus import NeptuneNexus
import traceback

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
    parser = argparse.ArgumentParser(description='Train Ghostbuster model using new architecture.')
    parser.add_argument('--dataset_handle', type=str, default='anakib1/mango-truth',
                       help='The dataset handle to use.')
    parser.add_argument('--dataset_config', type=str, default='xlsum',
                       help='The dataset configuration.')
    parser.add_argument('--tokenizer_handle', type=str, default='gugarosa/cl100k_base',
                       help='Tokenizer for n-gram models.')
    parser.add_argument('--llm_handles', type=str, nargs='+', default=['babbage-002'],
                       help='LLM model handles for probability estimation.')
    parser.add_argument('--human_samples', type=int, default=1000,
                       help='Number of human samples.')
    parser.add_argument('--ai_samples', type=int, default=1000,
                       help='Number of AI samples.')
    parser.add_argument('--max_length', type=int, default=15000,
                       help='Maximum text length for processing.')
    parser.add_argument('--validation_split', type=float, default=0.3,
                       help='Validation split ratio.')
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter for logistic regression.')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='Maximum iterations for logistic regression.')
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Output directory for training artifacts.')
    parser.add_argument('--save_model_path', type=str, default='./models/ghostbuster_detector',
                       help='Path to save the final trained model.')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Ghostbuster training with arguments: {args}")
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset_handle} with config {args.dataset_config}")
    hf_data = load_dataset(args.dataset_handle, args.dataset_config)
    
    # Process training data
    logger.info("Processing training data...")
    train_dataset = process_huggingface_dataset(
        hf_data['train'], 
        max(args.human_samples, args.ai_samples)
    )
    
    # Process test data if available
    test_dataset = None
    if 'test' in hf_data:
        logger.info("Processing test data...")
        test_dataset = process_huggingface_dataset(
            hf_data['test'], 
            min(500, max(args.human_samples, args.ai_samples) // 2)
        )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}")
    if test_dataset:
        logger.info(f"Test: {len(test_dataset)}")
    
    # Filter by length
    logger.info(f"Filtering samples by max length: {args.max_length}")
    initial_train_size = len(train_dataset)
    train_dataset = Dataset.from_samples([
        sample for sample in train_dataset 
        if len(sample.output) < args.max_length
    ])
    logger.info(f"Train dataset reduced from {initial_train_size} to {len(train_dataset)} after filtering")
    
    if test_dataset:
        initial_test_size = len(test_dataset)
        test_dataset = Dataset.from_samples([
            sample for sample in test_dataset 
            if len(sample.output) < args.max_length
        ])
        logger.info(f"Test dataset reduced from {initial_test_size} to {len(test_dataset)} after filtering")
    
    # Create trainable model
    logger.info("Initializing Ghostbuster model")
    model = TrainableGhostbusterDetector()
    
    # Create run identifier
    run_id = uuid4()
    run_surname = f"ghostbuster-{args.tokenizer_handle.replace('/', '-')}-{args.human_samples}-{run_id.hex[:8]}"
    
    # Create training configuration
    training_config = GhostbusterTrainingConfig(
        tokenizer_handle=args.tokenizer_handle,
        llm_handles=args.llm_handles,
        max_length=args.max_length,
        validation_split=args.validation_split,
        classifier_config={
            'C': args.C,
            'max_iter': args.max_iter
        },
        output_dir=f"{args.output_dir}/{run_surname}",
        run_id=run_id
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = GhostbusterTrainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    logger.info("Note: This may take a while due to feature extraction...")
    result = trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    # Print results
    logger.info(f"Training completed in {result.training_duration_seconds:.2f} seconds")
    logger.info(f"Total samples processed: {result.metadata['total_samples']}")
    logger.info(f"Tokenizer: {result.metadata['tokenizer_handle']}")
    logger.info(f"LLM handles: {result.metadata['llm_handles']}")
    
    # Print metrics
    if result.validation_conclusion:
        val_metrics = result.validation_conclusion
        logger.info(f"Validation Metrics:")
        logger.info(f"  Accuracy: {val_metrics.accuracy:.4f}")
        logger.info(f"  F1: {val_metrics.f1:.4f}")
        logger.info(f"  AUC: {val_metrics.auc:.4f}")
        logger.info(f"  Precision: {val_metrics.precision:.4f}")
        logger.info(f"  Recall: {val_metrics.recall:.4f}")
    
    if result.test_conclusion:
        test_metrics = result.test_conclusion
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
        
        # Create proper conclusion structure for Neptune
        from detectors.metrics import SplitConclusion
        
        # Create SplitConclusion objects (without representations for now)
        train_split_conclusion = SplitConclusion(
            metrics=result.train_conclusion,
            representations=None
        )
        
        validation_split_conclusion = None
        if result.validation_conclusion:
            validation_split_conclusion = SplitConclusion(
                metrics=result.validation_conclusion,
                representations=None
            )
        
        # Create proper Conclusion object
        from detectors.metrics import Conclusion
        conclusion = Conclusion(
            weights=result.model_weights,
            detector_handle="ghostbuster",
            datasets=[args.dataset_config],
            train_conclusion=train_split_conclusion,
            validation_conclusion=validation_split_conclusion
        )
        
        extra_data = {
            "tokenizer_handle": args.tokenizer_handle,
            "llm_handles": args.llm_handles,
            "human_samples": args.human_samples,
            "ai_samples": args.ai_samples,
            "max_length": args.max_length,
            "validation_split": args.validation_split,
            "training_duration": result.training_duration_seconds,
            "total_samples": result.metadata['total_samples'],
            "C": args.C,
            "max_iter": args.max_iter,
        }
        
        nexus.conclude_run(run_id, conclusion=conclusion, extra_data=extra_data)
        logger.info(f"Successfully uploaded run {run_id} to Neptune")
        
    except Exception as e:
        logger.warning(f"Failed to upload to Neptune: {traceback.format_exc()}")
    
    # Test loading the saved model
    logger.info("Testing model loading...")
    loaded_model = TrainableGhostbusterDetector()
    trainer_test = GhostbusterTrainer(loaded_model, training_config)
    trainer_test.load_model(args.save_model_path)
    
    # Test prediction
    test_text = "Artificial intelligence is transforming how we interact with technology."
    proba = loaded_model.predict_proba(test_text)
    labels = loaded_model.get_labels()
    
    logger.info(f"Test prediction for: '{test_text}'")
    for i, label in enumerate(labels):
        logger.info(f"  {label}: {proba[i]:.4f}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    load_dotenv()
    main() 