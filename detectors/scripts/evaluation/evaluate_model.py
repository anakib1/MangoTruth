"""Universal model evaluation script.

This script provides standalone model evaluation capabilities, completely
separated from training logic while using the same centralized evaluation
components that trainers use.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import yaml
from datasets import load_dataset

from detectors.models.zoo.registry import ModelRegistry
from detectors.evaluation import StandardEvaluator, EvaluationResult
from detectors.training.preprocessing import DatasetProcessorFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_path(model_path: str, model_type: str = 'auto'):
    """Load a model from a saved path.
    
    Args:
        model_path: Path to the saved model.
        model_type: Type of model to load ('auto', 'huggingface', etc.).
        
    Returns:
        Loaded model instance.
    """
    if model_type == 'auto':
        # Try to detect model type from path/metadata
        metadata_path = Path(model_path) / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Extract model type from metadata
            model_type = metadata.get('config', {}).get('model_name', 'huggingface')
    
    # Use model registry to load appropriate model
    registry = ModelRegistry()
    return registry.load_model(model_path, model_type)


def evaluate_on_datasets(evaluator: StandardEvaluator,
                         model,
                         datasets: Dict[str, Any],
                         batch_size: int = 32,
                         save_results: bool = True,
                         output_dir: Optional[str] = None) -> Dict[str, EvaluationResult]:
    """Evaluate model on multiple datasets.
    
    Args:
        evaluator: Evaluator instance.
        model: Model to evaluate.
        datasets: Dictionary of dataset names to datasets.
        batch_size: Batch size for evaluation.
        save_results: Whether to save results to files.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary of evaluation results.
    """
    results = evaluator.batch_evaluate(model, datasets, batch_size)
    
    # Save results if requested
    if save_results and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, result in results.items():
            # Save metrics as JSON
            metrics_file = output_path / f"{dataset_name}_metrics.json"
            metrics_data = {
                'accuracy': result.conclusion.metrics.accuracy,
                'f1': result.conclusion.metrics.f1,
                'auc': result.conclusion.metrics.auc,
                'precision': result.conclusion.metrics.precision,
                'recall': result.conclusion.metrics.recall,
                'dataset_info': result.dataset_info,
                'model_info': result.model_info,
                'evaluation_config': result.evaluation_config
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Saved metrics for {dataset_name} to {metrics_file}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Universal model evaluation script.')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model.')
    parser.add_argument('--model_type', type=str, default='auto',
                       help='Type of model (auto, huggingface, etc.).')
    
    # Dataset configuration
    parser.add_argument('--dataset_config', type=str,
                       help='Path to dataset configuration file.')
    parser.add_argument('--dataset_handle', type=str,
                       help='HuggingFace dataset handle.')
    parser.add_argument('--dataset_name', type=str,
                       help='Dataset configuration name.')
    parser.add_argument('--test_size', type=int, default=1000,
                       help='Number of test samples per class.')
    
    # Evaluation configuration
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation.')
    parser.add_argument('--include_visualizations', action='store_true',
                       help='Include ROC curves and confusion matrices.')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold.')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results.')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions.')
    
    # Multiple dataset evaluation
    parser.add_argument('--multiple_datasets', type=str, nargs='+',
                       help='List of dataset configurations to evaluate on.')
    
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation of model: {args.model_path}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model_from_path(args.model_path, args.model_type)
    logger.info(f"Loaded model: {model.__class__.__name__}")
    logger.info(f"Model labels: {model.get_labels()}")
    
    # Create evaluator
    evaluator = StandardEvaluator()
    
    # Prepare datasets
    datasets = {}
    
    if args.dataset_config:
        # Load from configuration file
        with open(args.dataset_config, 'r') as f:
            dataset_configs = yaml.safe_load(f)
        
        for dataset_name, config in dataset_configs.items():
            logger.info(f"Loading dataset: {dataset_name}")
            
            # Load HuggingFace dataset
            hf_data = load_dataset(config['handle'], config['config'])
            
            # Process dataset
            processor_config = config.get('processor', {})
            processor = DatasetProcessorFactory.create_processor('huggingface', **processor_config)
            
            test_size = config.get('test_size', args.test_size)
            dataset = processor.process(hf_data['test'], test_size, seed=42)
            datasets[dataset_name] = dataset
    
    elif args.dataset_handle:
        # Single dataset from command line
        logger.info(f"Loading dataset: {args.dataset_handle}:{args.dataset_name}")
        hf_data = load_dataset(args.dataset_handle, args.dataset_name)
        
        processor = DatasetProcessorFactory.create_processor('huggingface')
        dataset = processor.process(hf_data['test'], args.test_size, seed=42)
        datasets['test'] = dataset
    
    else:
        raise ValueError("Must specify either --dataset_config or --dataset_handle")
    
    # Perform evaluation
    logger.info(f"Evaluating on {len(datasets)} dataset(s)")
    
    evaluation_kwargs = {
        'batch_size': args.batch_size,
        'threshold': args.threshold,
        'include_visualizations': args.include_visualizations
    }
    
    results = evaluate_on_datasets(
        evaluator=evaluator,
        model=model,
        datasets=datasets,
        batch_size=args.batch_size,
        save_results=True,
        output_dir=args.output_dir
    )
    
    # Print summary results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*50)
    
    for dataset_name, result in results.items():
        metrics = result.conclusion.metrics
        logger.info(f"\nDataset: {dataset_name}")
        logger.info(f"  Samples: {result.dataset_info['size']}")
        logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"  F1 Score: {metrics.f1:.4f}")
        logger.info(f"  AUC: {metrics.auc:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
    
    # Save individual predictions if requested
    if args.save_predictions:
        predictions_dir = Path(args.output_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, result in results.items():
            pred_file = predictions_dir / f"{dataset_name}_predictions.json"
            pred_data = {
                'predictions': result.predictions.tolist(),
                'true_labels': result.true_labels.tolist(),
                'dataset_info': result.dataset_info
            }
            
            with open(pred_file, 'w') as f:
                json.dump(pred_data, f, indent=2)
            
            logger.info(f"Saved predictions for {dataset_name} to {pred_file}")
    
    logger.info(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main() 