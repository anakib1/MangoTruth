"""RoBERTa training script with HuggingFace Nexus upload.

This script trains a RoBERTa model for AI text detection on the XLSum dataset
and uploads the trained model to HuggingFace Hub using HuggingFaceNexus.
"""

import argparse
import logging
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from detectors.data.datasets import Dataset
from detectors.data.datasets.loaders import HuggingFaceLoader
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.hub_nexus import HuggingFaceNexus
from detectors.models.zoo.registry import ModelRegistry
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.training import HuggingFaceTrainer, HuggingFaceTrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RoBERTa model on XLSum and upload to HuggingFace Hub.')

    # Dataset configuration
    parser.add_argument('--dataset_handle', type=str, default='anakib1/mango-truth',
                        help='The dataset handle to use.')
    parser.add_argument('--dataset_config', type=str, default='xlsum',
                        help='The dataset configuration.')

    # Model configuration
    parser.add_argument('--model_handle', type=str, default='FacebookAI/roberta-base',
                        help='The model handle to use.')

    # Data size configuration
    parser.add_argument('--train_size', type=int, default=5000,
                        help='Number of training examples per class.')
    parser.add_argument('--test_size', type=int, default=1000,
                        help='Number of test examples per class.')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for both training and evaluation.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer.')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./training_output',
                        help='Output directory for training artifacts.')
    parser.add_argument('--save_model_path', type=str, default='./models/roberta_xlsum_detector',
                        help='Path to save the final trained model.')

    # HuggingFace Hub configuration
    parser.add_argument('--hf_namespace', type=str, default='anakib1',
                        help='HuggingFace namespace/organization for model upload.')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API token. If not provided, will try to get from environment.')
    parser.add_argument('--hf_private', action='store_true',
                        help='Create private repository on HuggingFace Hub.')
    parser.add_argument('--skip_hf_upload', action='store_true',
                        help='Skip uploading to HuggingFace Hub.')

    # Training efficiency
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision training.')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience.')

    args = parser.parse_args()

    logger.info(f"Starting RoBERTa training with XLSum dataset")
    logger.info(f"Model: {args.model_handle}")
    logger.info(f"Dataset: {args.dataset_handle}:{args.dataset_config}")
    logger.info(f"Training samples: {args.train_size} per class")
    logger.info(f"Test samples: {args.test_size} per class")

    # Load dataset
    logger.info(f"Loading dataset {args.dataset_handle} with config {args.dataset_config}")
    logger.info("Creating dataset processor...")

    # Process training data with validation split
    logger.info("Processing training data...")

    loader = HuggingFaceLoader(args.dataset_handle, args.dataset_config, author_column='user_id')
    dataset = Dataset(loader=loader).filter(lambda x: len(x.output) < 1000)

    total = args.train_size + args.test_size
    train_dataset, val_dataset = dataset.split([args.train_size / total, args.test_size / total], seed=42)
    val_dataset, test_dataset = val_dataset.split([0.5, 0.5], seed=42)

    # Create model configuration
    model_config = HuggingFaceConfig(
        model_name=args.model_handle,
        tokenizer_name=args.model_handle,
        num_labels=2,
        max_length=512,
        cache_dir="./model_cache"
    )

    logger.info(f"Initializing model {args.model_handle}...")
    model = TrainableHuggingFaceDetector(model_config)

    # Update model's label mapping
    model.model.config.id2label = {0: "human", 1: "gpt-4o-mini"}
    model.model.config.label2id = {v: k for k, v in model.model.config.id2label.items()}

    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Trainable parameters: {model.get_num_trainable_parameters():,}")

    # Create a run identifier
    run_id = uuid4()
    model_name_short = args.model_handle.split('/')[-1]
    run_surname = f"roberta-xlsum-{model_name_short}-{args.train_size}-{run_id.hex[:8]}"

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
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        # Optimization
        optimizer_name="adamw_torch",
        scheduler_type="linear",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,

        # Evaluation and saving
        validation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,

        # Output
        output_dir=f"{args.output_dir}/{run_surname}",

        # Training efficiency
        fp16=args.fp16,
        dataloader_num_workers=0,

        # Early stopping
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,

        # Metrics
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # HuggingFace Hub
        push_to_hub=False,  # We'll handle upload manually with HuggingFaceNexus

        # Run ID
        run_id=run_id
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = HuggingFaceTrainer(model, training_config)

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
        logger.info(f"  Precision: {val_metrics.precision:.4f}")
        logger.info(f"  Recall: {val_metrics.recall:.4f}")

    if result.test_conclusion:
        test_metrics = result.test_conclusion.train_conclusion.metrics
        logger.info("Test Metrics:")
        logger.info(f"  Accuracy: {test_metrics.accuracy:.4f}")
        logger.info(f"  F1: {test_metrics.f1:.4f}")
        logger.info(f"  AUC: {test_metrics.auc:.4f}")
        logger.info(f"  Precision: {test_metrics.precision:.4f}")
        logger.info(f"  Recall: {test_metrics.recall:.4f}")

    # Save the final model locally
    Path(args.save_model_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {args.save_model_path}/model.pt")
    trainer.save_model(args.save_model_path +"/model.pt" )

    # Upload to HuggingFace Hub using HuggingFaceNexus
    if not args.skip_hf_upload:
        try:
            logger.info("Uploading model to HuggingFace Hub...")

            # Initialize HuggingFace Nexus
            hf_nexus = HuggingFaceNexus(
                namespace=args.hf_namespace,
                token=args.hf_token,
                private=args.hf_private
            )

            # Create a conclusion for upload
            conclusion = result.train_conclusion

            # Add extra metadata
            extra_data = {
                "model_handle": args.model_handle,
                "dataset_handle": args.dataset_handle,
                "dataset_config": args.dataset_config,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "training_duration": result.training_duration_seconds,
                "total_params": model.get_num_parameters(),
                "trainable_params": model.get_num_trainable_parameters(),
                "best_checkpoint": result.best_checkpoint_path,
                "validation_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "train_samples": len(train_dataset)
            }

            # Upload to HuggingFace Hub
            hf_nexus.conclude_run(run_id, conclusion=conclusion, extra_data=extra_data)

            repo_id = hf_nexus._get_repo_id(run_id)
            logger.info(f"Successfully uploaded model to HuggingFace Hub: {repo_id}")

            # Test loading the model back from HuggingFace Hub
            logger.info("Testing model loading from HuggingFace Hub...")

            # Create a model registry with HuggingFace Nexus
            registry = ModelRegistry(nexus=hf_nexus)

            # Load the model using the run ID
            loaded_model = registry.load_pretrained_model(str(run_id))

            # Test prediction
            test_text = "Artificial intelligence is revolutionizing healthcare by enabling more accurate diagnoses and personalized treatment plans."
            probs = loaded_model.predict_proba(test_text)
            labels = loaded_model.get_labels()

            logger.info(f"Test prediction for: '{test_text}'")
            for label, prob in zip(labels, probs):
                logger.info(f"  {label}: {prob:.4f}")

            logger.info(f"Model successfully uploaded and verified at: https://huggingface.co/{repo_id}")

        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace Hub: {e}")
            logger.info("Model training completed but upload failed. Model saved locally.")

    else:
        logger.info("Skipping HuggingFace Hub upload as requested.")

        # Still test the model locally
        test_text = "Artificial intelligence is revolutionizing healthcare by enabling more accurate diagnoses and personalized treatment plans."
        probs = model.predict_proba(test_text)
        labels = model.get_labels()

        logger.info(f"Test prediction for: '{test_text}'")
        for label, prob in zip(labels, probs):
            logger.info(f"  {label}: {prob:.4f}")

    logger.info("Training completed successfully!")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model saved locally at: {args.save_model_path}")


if __name__ == '__main__':
    load_dotenv()
    main()
