import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from detectors.data.datasets.folder_dataset import FolderDataset
from detectors.data.datasets.huggingface_dataset import HuggingFaceDataset
from detectors.data.generators.llm_generator import LLMGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Example 1: Load data from folder structure
    folder_dataset = FolderDataset("path/to/your/data")
    folder_dataset.load()
    logger.info(f"Loaded {len(folder_dataset)} samples from folder")
    
    # Print unique labels in the dataset
    unique_labels = folder_dataset.get_unique_labels()
    logger.info(f"Found labels in dataset: {unique_labels}")
    
    # Example 2: Load data from HuggingFace with different split options
    # Option 1: Load single split
    train_dataset = HuggingFaceDataset(
        dataset_name="your-dataset-name",
        split="train",
        prompt_column="input",
        output_column="output",
        author_column="author",
        label_column="label"
    )
    train_dataset.load()
    logger.info(f"Loaded {len(train_dataset)} samples from train split")
    
    # Option 2: Load multiple splits at once
    multi_split_dataset = HuggingFaceDataset(
        dataset_name="your-dataset-name",
        split=["train", "validation", "test"],
        prompt_column="input",
        output_column="output",
        author_column="author",
        label_column="label"
    )
    multi_split_dataset.load()
    logger.info(f"Loaded {len(multi_split_dataset)} samples from multiple splits")
    
    # Print available splits and their sizes
    available_splits = multi_split_dataset.get_available_splits()
    for split in available_splits:
        split_samples = multi_split_dataset.get_samples_by_split(split)
        logger.info(f"Split '{split}': {len(split_samples)} samples")
        
        # Print label distribution in this split
        split_labels = set(sample.label for sample in split_samples)
        for label in split_labels:
            label_samples = [s for s in split_samples if s.label == label]
            logger.info(f"  - {label}: {len(label_samples)} samples")
    
    # Example 3: Generate synthetic data using different models
    generator = LLMGenerator(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Generate samples with GPT-4
    gpt4_dataset = await generator.generate_dataset(
        source_dataset=folder_dataset,
        generation_type="rewrite",
        num_samples=10,
        model_label="gpt-4"
    )
    logger.info(f"Generated {len(gpt4_dataset)} samples with GPT-4")
    
    # Generate samples with Claude (if available)
    claude_dataset = await generator.generate_dataset(
        source_dataset=folder_dataset,
        generation_type="variation",
        num_samples=5,
        model_label="claude"
    )
    logger.info(f"Generated {len(claude_dataset)} samples with Claude")
    
    # Generate multiple variations with different models
    variations_dataset = await generator.generate_multiple_variations(
        source_dataset=folder_dataset,
        num_variations=3,
        model_label="gpt-4",
        generation_type="variation",
        num_samples=5
    )
    logger.info(f"Generated {len(variations_dataset)} variation samples")
    
    # Save generated datasets
    output_dir = Path("generated_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save each model's samples separately
    for model_label in ["gpt-4", "claude"]:
        model_samples = [s for s in variations_dataset.samples if s.label.startswith(model_label)]
        if model_samples:
            model_dataset = BaseDataset()
            model_dataset.samples = model_samples
            model_dataset.save_samples(output_dir / model_label)
    
    # Split dataset for training
    train_val_test = folder_dataset.split(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    for split_name, split_dataset in train_val_test.items():
        logger.info(f"{split_name} split: {len(split_dataset)} samples")
        split_dataset.save_samples(output_dir / split_name)
        
        # Print label distribution in each split
        for label in split_dataset.get_unique_labels():
            label_samples = split_dataset.get_samples_by_label(label)
            logger.info(f"{split_name} split - {label}: {len(label_samples)} samples")

if __name__ == "__main__":
    asyncio.run(main()) 