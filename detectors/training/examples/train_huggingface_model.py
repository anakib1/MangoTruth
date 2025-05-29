"""Example script for training a HuggingFace plagiarism detector.

This script demonstrates how to use the training pipeline to train
a HuggingFace sequence classification model for plagiarism detection.
"""

import logging
from pathlib import Path

from detectors.data.datasets import Dataset, TextSample
from detectors.models.zoo.configs import HuggingFaceConfig
from detectors.models.zoo.trainable_implementations import TrainableHuggingFaceDetector
from detectors.training import HuggingFaceTrainer, HuggingFaceTrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_example_dataset():
    """Create a small example dataset for demonstration."""
    samples = [
        # Human-written samples
        TextSample(
            prompt="Write about climate change",
            output="Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures are causing widespread environmental disruption.",
            author_id="human_1",
            label="human"
        ),
        TextSample(
            prompt="Describe machine learning",
            output="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed for every scenario.",
            author_id="human_2",
            label="human"
        ),
        TextSample(
            prompt="Explain quantum computing",
            output="Quantum computing harnesses the principles of quantum mechanics to process information in fundamentally different ways than classical computers.",
            author_id="human_3",
            label="human"
        ),
        # AI-generated samples
        TextSample(
            prompt="Write about climate change",
            output="Climate change is a significant global issue that requires immediate attention. The Earth's temperature is rising due to greenhouse gas emissions.",
            author_id="gpt-4",
            label="ai"
        ),
        TextSample(
            prompt="Describe machine learning",
            output="Machine learning is a branch of AI that focuses on building systems that learn from data. These systems improve their performance through experience.",
            author_id="gpt-4",
            label="ai"
        ),
        TextSample(
            prompt="Explain quantum computing",
            output="Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be infeasible for classical computers.",
            author_id="claude",
            label="ai"
        ),
    ]
    
    return Dataset.from_samples(samples)


def main():
    """Main training function."""
    
    # Create dataset
    logger.info("Creating example dataset...")
    dataset = create_example_dataset()
    
    # Split into train/validation
    train_dataset, val_dataset = dataset.split([0.8, 0.2], seed=42)
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create model configuration
    model_config = HuggingFaceConfig(
        model_name="distilbert-base-uncased",  # Using DistilBERT for efficiency
        tokenizer_name="distilbert-base-uncased",
        num_labels=2,  # Binary classification: human vs ai
        max_length=256,
        cache_dir="./model_cache"
    )
    
    # Create trainable model
    logger.info("Initializing model...")
    model = TrainableHuggingFaceDetector(model_config)
    
    # Update model's label mapping
    model.model.config.id2label = {0: "human", 1: "ai"}
    model.model.config.label2id = {"human": 0, "ai": 1}
    
    # Create training configuration
    training_config = HuggingFaceTrainingConfig(
        # Model info
        model_name="distilbert-base-uncased",
        num_labels=2,
        max_length=256,
        
        # Training parameters
        epochs=3,
        batch_size=2,  # Small batch size for example
        learning_rate=5e-5,
        warmup_ratio=0.1,
        
        # Evaluation
        validation_strategy="epoch",
        save_strategy="epoch",
        
        # Output
        output_dir="./training_output/distilbert_plagiarism",
        
        # Efficiency
        fp16=False,  # Disable for CPU training
        dataloader_num_workers=0,
        
        # Early stopping
        early_stopping=True,
        early_stopping_patience=2,
        
        # Metrics
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = HuggingFaceTrainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    result = trainer.train(
        train_dataset=train_dataset,
        validation_dataset=val_dataset
    )
    
    # Print results
    logger.info(f"Training completed in {result.training_duration_seconds:.2f} seconds")
    logger.info(f"Best checkpoint: {result.best_checkpoint_path}")
    
    # Print final metrics
    if result.validation_conclusion:
        val_metrics = result.validation_conclusion.metrics
        logger.info(f"Validation Accuracy: {val_metrics.accuracy:.4f}")
        logger.info(f"Validation F1: {val_metrics.f1:.4f}")
        logger.info(f"Validation AUC: {val_metrics.auc:.4f}")
    
    # Save final model
    final_model_path = "./models/distilbert_plagiarism_detector"
    trainer.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    # Example of loading and using the trained model
    logger.info("Loading trained model...")
    loaded_model = TrainableHuggingFaceDetector.from_pretrained(final_model_path)
    
    # Test prediction
    test_text = "Artificial intelligence is transforming how we interact with technology."
    probs = loaded_model.predict_proba(test_text)
    labels = loaded_model.get_labels()
    
    logger.info(f"Test prediction for: '{test_text}'")
    for label, prob in zip(labels, probs):
        logger.info(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main() 