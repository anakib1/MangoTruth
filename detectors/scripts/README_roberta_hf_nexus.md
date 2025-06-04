# RoBERTa Training with HuggingFace Nexus Upload

This script (`train_roberta_hf_nexus.py`) trains a RoBERTa model for AI text detection on the XLSum dataset and uploads the trained model to HuggingFace Hub using the HuggingFaceNexus integration.

## Features

- **Model**: RoBERTa-base for sequence classification
- **Dataset**: XLSum dataset (news/article continuations) from `anakib1/mango-truth`
- **Task**: Binary classification (human vs AI-generated text)
- **Upload**: Automatic upload to HuggingFace Hub with model card generation
- **Evaluation**: Comprehensive metrics on train/validation/test splits

## Prerequisites

1. Install required dependencies:
   ```bash
   pip install transformers datasets huggingface_hub python-dotenv
   ```

2. Set up HuggingFace token (optional for upload):
   ```bash
   # Option 1: Environment variable
   export HF_TOKEN="your_huggingface_token_here"
   
   # Option 2: .env file
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   
   # Option 3: Pass via command line
   python train_roberta_hf_nexus.py --hf_token "your_token_here"
   ```

## Basic Usage

### Quick Start
```bash
python train_roberta_hf_nexus.py
```

### With Custom Parameters
```bash
python train_roberta_hf_nexus.py \
    --model_handle "FacebookAI/roberta-base" \
    --train_size 3000 \
    --test_size 500 \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --hf_namespace "YourOrganization" \
    --hf_private
```

### Skip HuggingFace Upload (Local Training Only)
```bash
python train_roberta_hf_nexus.py \
    --skip_hf_upload \
    --train_size 1000 \
    --epochs 3
```

## Command Line Arguments

### Dataset Configuration
- `--dataset_handle`: HuggingFace dataset handle (default: `anakib1/mango-truth`)
- `--dataset_config`: Dataset configuration (default: `xlsum`)
- `--train_size`: Number of training examples per class (default: 5000)
- `--test_size`: Number of test examples per class (default: 1000)

### Model Configuration
- `--model_handle`: HuggingFace model handle (default: `FacebookAI/roberta-base`)

### Training Configuration
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--warmup_ratio`: Warmup ratio for LR scheduler (default: 0.1)
- `--weight_decay`: Weight decay (default: 0.01)
- `--fp16`: Enable mixed precision training (default: True)
- `--early_stopping_patience`: Early stopping patience (default: 3)

### Output Configuration
- `--output_dir`: Training output directory (default: `./training_output`)
- `--save_model_path`: Local model save path (default: `./models/roberta_xlsum_detector`)

### HuggingFace Hub Configuration
- `--hf_namespace`: HuggingFace namespace/organization (default: `MangoTruth`)
- `--hf_token`: HuggingFace API token
- `--hf_private`: Create private repository
- `--skip_hf_upload`: Skip uploading to HuggingFace Hub

## Output

### Local Files
- **Model**: Saved to `--save_model_path`
- **Training logs**: Console output with detailed metrics
- **Checkpoints**: Saved to `--output_dir/{run_surname}`

### HuggingFace Hub
- **Repository**: `{namespace}/plagiarism-detector-{run_id}`
- **Model weights**: `model_weights.pkl`
- **Metadata**: `run_metadata.json`
- **Model card**: `README.md` with metrics and usage instructions

## Example Output

```
2024-01-15 10:30:00 - __main__ - INFO - Starting RoBERTa training with XLSum dataset
2024-01-15 10:30:00 - __main__ - INFO - Model: FacebookAI/roberta-base
2024-01-15 10:30:00 - __main__ - INFO - Dataset: anakib1/mango-truth:xlsum
...
2024-01-15 10:45:00 - __main__ - INFO - Validation Metrics:
2024-01-15 10:45:00 - __main__ - INFO -   Accuracy: 0.8642
2024-01-15 10:45:00 - __main__ - INFO -   F1: 0.8598
...
2024-01-15 10:45:30 - __main__ - INFO - Successfully uploaded model to HuggingFace Hub: MangoTruth/plagiarism-detector-abc123...
2024-01-15 10:45:35 - __main__ - INFO - Model successfully uploaded and verified at: https://huggingface.co/MangoTruth/plagiarism-detector-abc123...
```

## Using the Trained Model

After training and upload, you can load the model using the ModelRegistry:

```python
from detectors.models.zoo import ModelRegistry
from detectors.models.zoo.hub_nexus import HuggingFaceNexus

# Initialize registry with HuggingFace Nexus
nexus = HuggingFaceNexus(namespace="MangoTruth")
registry = ModelRegistry(nexus=nexus)

# Load by run ID
model = registry.load_pretrained_model("your-run-id-here")

# Make predictions
text = "Your text to classify..."
probabilities = model.predict_proba(text)
labels = model.get_labels()

for label, prob in zip(labels, probabilities):
    print(f"{label}: {prob:.4f}")
```

## Performance Notes

- **Training time**: ~15-30 minutes for 5K samples per class on GPU
- **Memory usage**: ~8GB GPU memory with batch size 64
- **Model size**: ~125M parameters (RoBERTa-base)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` (try 32 or 16)
2. **No HuggingFace token**: Use `--skip_hf_upload` for local-only training
3. **Dataset loading fails**: Check internet connection and dataset availability
4. **Upload permission denied**: Verify HuggingFace token has write permissions

### Debug Mode
Add more verbose logging:
```bash
export TRANSFORMERS_VERBOSITY=info
python train_roberta_hf_nexus.py --batch_size 16
``` 