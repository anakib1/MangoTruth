"""Tests for centralized evaluation components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from detectors.evaluation.evaluators import StandardEvaluator, evaluate_model
from detectors.evaluation.interfaces import EvaluationResult
from detectors.evaluation.metrics import MetricsCalculator
from detectors.metrics import SplitConclusion, ClassificationMetrics


class TestStandardEvaluator:
    """Tests for StandardEvaluator class."""
    
    def test_initialization_default(self):
        """Test default initialization of StandardEvaluator."""
        evaluator = StandardEvaluator()
        
        assert evaluator.metrics_calculator is None
        assert evaluator.label_mapping_strategy == 'auto'
    
    def test_initialization_custom(self):
        """Test custom initialization with parameters."""
        custom_calculator = Mock()
        evaluator = StandardEvaluator(
            metrics_calculator=custom_calculator,
            label_mapping_strategy='manual'
        )
        
        assert evaluator.metrics_calculator == custom_calculator
        assert evaluator.label_mapping_strategy == 'manual'
    
    def test_create_label_mapping_exact_match(self, sample_dataset, mock_detector):
        """Test label mapping with exact matches."""
        evaluator = StandardEvaluator()
        
        mapping = evaluator._create_label_mapping(sample_dataset, mock_detector)
        
        # Should map directly for exact matches
        assert mapping["human"] == 0  # "human" maps to index 0
        assert mapping["ai"] == 1     # "ai" maps to index 1
    
    def test_create_label_mapping_case_insensitive(self, mock_detector):
        """Test label mapping with case-insensitive matching."""
        # Create dataset with different case
        from detectors.data.datasets import Dataset, TextSample
        samples = [
            TextSample("prompt", "text1", "author1", "Human"),  # Capital H
            TextSample("prompt", "text2", "author2", "AI")      # Capital AI
        ]
        dataset = Dataset.from_samples(samples)
        
        evaluator = StandardEvaluator()
        mapping = evaluator._create_label_mapping(dataset, mock_detector)
        
        # Should still map correctly despite case differences
        assert mapping["Human"] == 0
        assert mapping["AI"] == 1
    
    def test_create_label_mapping_semantic(self, mock_detector):
        """Test semantic label mapping for AI/human variants."""
        from detectors.data.datasets import Dataset, TextSample
        samples = [
            TextSample("prompt", "text1", "author1", "machine"),    # AI-related
            TextSample("prompt", "text2", "author2", "person")      # Human-related
        ]
        dataset = Dataset.from_samples(samples)
        
        evaluator = StandardEvaluator()
        mapping = evaluator._create_label_mapping(dataset, mock_detector)
        
        # Should map semantically
        assert mapping["machine"] == 1  # AI-related maps to "ai" index
        assert mapping["person"] == 0   # Human-related maps to "human" index
    
    def test_create_label_mapping_fallback(self, mock_detector):
        """Test label mapping fallback for unknown labels."""
        from detectors.data.datasets import Dataset, TextSample
        samples = [
            TextSample("prompt", "text1", "author1", "unknown_label")
        ]
        dataset = Dataset.from_samples(samples)
        
        evaluator = StandardEvaluator()
        mapping = evaluator._create_label_mapping(dataset, mock_detector)
        
        # Should fallback to index 0
        assert mapping["unknown_label"] == 0
    
    def test_prepare_predictions(self, mock_detector, sample_dataset):
        """Test prediction preparation."""
        evaluator = StandardEvaluator()
        
        # Override the mock batch predict with specific behavior for this test
        call_count = 0
        def mock_batch_predict(texts):
            nonlocal call_count
            batch_size = len(texts)
            call_count += 1
            # Return predictions for the batch size
            if batch_size == 2:
                if call_count == 1:
                    return np.array([[0.8, 0.2], [0.3, 0.7]])
                else:
                    return np.array([[0.9, 0.1], [0.4, 0.6]])
            else:
                return np.array([[0.5, 0.5]] * batch_size)
        
        mock_detector.batch_predict = mock_batch_predict
        
        texts = [sample.output for sample in sample_dataset]
        predictions = evaluator._prepare_predictions(mock_detector, texts, batch_size=2)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (4, 2)
        assert call_count == 2  # 2 batches of size 2
    
    def test_prepare_predictions_large_batch(self, mock_detector):
        """Test prediction preparation with large batch."""
        evaluator = StandardEvaluator()
        
        # Create many texts
        texts = [f"text_{i}" for i in range(100)]
        
        # Override mock predictions with call counting
        call_count = 0
        def mock_batch_predict(texts):
            nonlocal call_count
            call_count += 1
            batch_size = len(texts)
            return np.array([[0.5, 0.5]] * batch_size)
        
        mock_detector.batch_predict = mock_batch_predict
        
        predictions = evaluator._prepare_predictions(mock_detector, texts, batch_size=32)
        
        assert predictions.shape[0] == 100
        # Should make 4 calls: 3 full batches + 1 partial batch (100 = 32*3 + 4)
        assert call_count == 4
    
    def test_convert_to_binary_classification_binary_model(self, mock_detector):
        """Test binary classification conversion for binary model."""
        evaluator = StandardEvaluator()
        
        predictions = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
        true_labels = np.array([1, 1, 0])  # AI, AI, Human
        model_labels = ["human", "ai"]
        
        binary_pred, binary_true = evaluator._convert_to_binary_classification(
            predictions, true_labels, model_labels
        )
        
        # Should use AI class (index 1) as positive
        np.testing.assert_array_equal(binary_pred, [0.3, 0.8, 0.1])
        np.testing.assert_array_equal(binary_true, [1, 1, 0])
    
    def test_convert_to_binary_classification_multiclass_model(self, mock_detector):
        """Test binary classification conversion for multi-class model."""
        evaluator = StandardEvaluator()
        
        predictions = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.8, 0.1, 0.1]])
        true_labels = np.array([0, 1, 0])
        model_labels = ["class1", "class2", "class3"]
        
        binary_pred, binary_true = evaluator._convert_to_binary_classification(
            predictions, true_labels, model_labels
        )
        
        # Should convert to correct/incorrect binary classification
        predicted_labels = np.argmax(predictions, axis=1)  # [0, 1, 0]
        expected_binary_true = (predicted_labels == true_labels).astype(int)  # [1, 1, 1]
        
        np.testing.assert_array_equal(binary_true, expected_binary_true)
        assert len(binary_pred) == 3
    
    @patch('detectors.evaluation.evaluators.MetricsCalculatorFactory')
    def test_evaluate_full_workflow(self, mock_factory, mock_detector, sample_dataset):
        """Test complete evaluation workflow."""
        # Setup mocks
        mock_calculator = Mock()
        mock_conclusion = Mock()
        mock_calculator.calculate_metrics.return_value = mock_conclusion
        mock_factory.create_for_model.return_value = mock_calculator
        
        # Override batch_predict for this test
        def mock_batch_predict(texts):
            batch_size = len(texts)
            if batch_size == 2:
                return np.array([[0.8, 0.2], [0.3, 0.7]])
            elif batch_size == 4:
                return np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
            else:
                return np.array([[0.5, 0.5]] * batch_size)
        
        mock_detector.batch_predict = mock_batch_predict
        mock_detector.__class__.__name__ = "MockDetector"  # Avoid format string issues
        
        evaluator = StandardEvaluator()
        result = evaluator.evaluate(mock_detector, sample_dataset, batch_size=2)
        
        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.conclusion == mock_conclusion
        assert isinstance(result.predictions, np.ndarray)
        assert isinstance(result.true_labels, np.ndarray)
        assert isinstance(result.dataset_info, dict)
        assert isinstance(result.model_info, dict)
        assert isinstance(result.evaluation_config, dict)
        
        # Verify dataset info
        assert result.dataset_info['size'] == 4
        assert 'labels' in result.dataset_info
        assert 'label_mapping' in result.dataset_info
        
        # Verify model info
        assert result.model_info['class'] == 'MockDetector'
        assert 'labels' in result.model_info
        
        # Verify evaluation config
        assert result.evaluation_config['batch_size'] == 2
    
    def test_evaluate_with_custom_metrics_calculator(self, mock_detector, sample_dataset):
        """Test evaluation with custom metrics calculator."""
        custom_calculator = Mock()
        mock_conclusion = Mock()
        custom_calculator.calculate_metrics.return_value = mock_conclusion
        
        evaluator = StandardEvaluator(metrics_calculator=custom_calculator)
        
        mock_detector.batch_predict.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]
        ])
        
        result = evaluator.evaluate(mock_detector, sample_dataset)
        
        # Should use the custom calculator
        custom_calculator.calculate_metrics.assert_called_once()
        assert result.conclusion == mock_conclusion
    
    def test_batch_evaluate(self, mock_detector, sample_dataset):
        """Test batch evaluation on multiple datasets."""
        evaluator = StandardEvaluator()
        
        # Create multiple datasets
        datasets = {
            "train": sample_dataset,
            "test": sample_dataset
        }
        
        mock_detector.batch_predict.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]
        ])
        
        with patch.object(evaluator, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_evaluate.return_value = mock_result
            
            results = evaluator.batch_evaluate(mock_detector, datasets, batch_size=32)
        
        # Should call evaluate for each dataset
        assert mock_evaluate.call_count == 2
        assert "train" in results
        assert "test" in results
        assert results["train"] == mock_result
        assert results["test"] == mock_result
    
    def test_evaluate_with_kwargs(self, mock_detector, sample_dataset):
        """Test evaluation with additional kwargs."""
        evaluator = StandardEvaluator()
        
        mock_detector.batch_predict.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]
        ])
        
        with patch.object(evaluator, 'metrics_calculator') as mock_calculator:
            mock_calculator.calculate_metrics.return_value = Mock()
            
            result = evaluator.evaluate(
                mock_detector, 
                sample_dataset, 
                threshold=0.6,
                include_visualizations=False
            )
        
        # Check that kwargs were passed to metrics calculator
        mock_calculator.calculate_metrics.assert_called_once()
        call_kwargs = mock_calculator.calculate_metrics.call_args[1]
        assert call_kwargs['threshold'] == 0.6
        assert call_kwargs['include_visualizations'] == False


class TestEvaluateModelFunction:
    """Tests for the convenience evaluate_model function."""
    
    @patch('detectors.evaluation.evaluators.StandardEvaluator')
    def test_evaluate_model_convenience_function(self, mock_evaluator_class, mock_detector, sample_dataset):
        """Test the convenience evaluate_model function."""
        mock_evaluator = Mock()
        mock_result = Mock()
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator_class.return_value = mock_evaluator
        
        result = evaluate_model(mock_detector, sample_dataset, batch_size=64)
        
        # Should create evaluator and call evaluate
        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(
            mock_detector, sample_dataset, 64
        )
        assert result == mock_result


class TestStandardEvaluatorErrorHandling:
    """Tests for error handling in StandardEvaluator."""
    
    def test_evaluate_with_prediction_error(self, mock_detector, sample_dataset):
        """Test handling of prediction errors."""
        evaluator = StandardEvaluator()
        
        # Mock batch_predict to raise an error
        mock_detector.batch_predict.side_effect = Exception("Prediction failed")
        
        with pytest.raises(Exception, match="Prediction failed"):
            evaluator.evaluate(mock_detector, sample_dataset)
    
    def test_evaluate_with_empty_dataset(self, mock_detector):
        """Test evaluation with empty dataset."""
        from detectors.data.datasets import Dataset
        empty_dataset = Dataset.from_samples([])
        
        evaluator = StandardEvaluator()
        
        # Mock empty predictions
        mock_detector.batch_predict.return_value = np.array([]).reshape(0, 2)
        
        with patch('detectors.evaluation.evaluators.MetricsCalculatorFactory') as mock_factory:
            mock_calculator = Mock()
            mock_conclusion = Mock()
            mock_calculator.calculate_metrics.return_value = mock_conclusion
            mock_factory.create_for_model.return_value = mock_calculator
            
            result = evaluator.evaluate(mock_detector, empty_dataset)
        
        # Should handle gracefully
        assert isinstance(result, EvaluationResult)
        assert result.dataset_info['size'] == 0
    
    def test_evaluate_with_mismatched_labels(self, mock_detector):
        """Test evaluation with completely mismatched labels."""
        from detectors.data.datasets import Dataset, TextSample
        
        # Dataset with labels that don't match model at all
        samples = [
            TextSample("prompt", "text1", "author1", "positive"),
            TextSample("prompt", "text2", "author2", "negative")
        ]
        dataset = Dataset.from_samples(samples)
        
        # Model with different labels
        mock_detector.get_labels.return_value = ["class_a", "class_b"]
        mock_detector.batch_predict.return_value = np.array([[0.6, 0.4], [0.3, 0.7]])
        
        evaluator = StandardEvaluator()
        
        with patch('detectors.evaluation.evaluators.MetricsCalculatorFactory') as mock_factory:
            mock_calculator = Mock()
            mock_conclusion = Mock()
            mock_calculator.calculate_metrics.return_value = mock_conclusion
            mock_factory.create_for_model.return_value = mock_calculator
            
            result = evaluator.evaluate(mock_detector, dataset)
        
        # Should use fallback mapping (all to index 0)
        mapping = result.dataset_info['label_mapping']
        assert mapping["positive"] == 0
        assert mapping["negative"] == 0


class TestStandardEvaluatorIntegration:
    """Integration tests for StandardEvaluator."""
    
    def test_end_to_end_evaluation(self, mock_detector, sample_dataset):
        """Test complete end-to-end evaluation."""
        # Setup realistic mock behavior
        mock_detector.get_labels.return_value = ["human", "ai"]
        mock_detector.get_num_parameters.return_value = 110000000
        mock_detector.batch_predict.return_value = np.array([
            [0.9, 0.1],  # Strong human prediction
            [0.2, 0.8],  # Strong AI prediction
            [0.8, 0.2],  # Human prediction
            [0.3, 0.7]   # AI prediction
        ])
        
        # Expected true labels: [human, ai, human, ai] -> [0, 1, 0, 1]
        evaluator = StandardEvaluator()
        
        # Use real metrics calculator for integration test
        result = evaluator.evaluate(mock_detector, sample_dataset, batch_size=2)
        
        # Verify result structure and content
        assert isinstance(result, EvaluationResult)
        assert result.predictions.shape == (4,)  # Binary predictions
        assert result.true_labels.shape == (4,)
        
        # Check that metrics were calculated
        assert hasattr(result.conclusion, 'metrics')
        
        # Check dataset info
        assert result.dataset_info['size'] == 4
        assert result.dataset_info['labels'] == ["human", "ai"]
        
        # Check model info
        assert result.model_info['labels'] == ["human", "ai"]
        assert result.model_info['parameters'] == 110000000
    
    def test_evaluation_consistency(self, mock_detector, sample_dataset):
        """Test that multiple evaluations of the same data give consistent results."""
        mock_detector.batch_predict.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]
        ])
        
        evaluator = StandardEvaluator()
        
        # Run evaluation twice
        result1 = evaluator.evaluate(mock_detector, sample_dataset)
        result2 = evaluator.evaluate(mock_detector, sample_dataset)
        
        # Results should be identical
        np.testing.assert_array_equal(result1.predictions, result2.predictions)
        np.testing.assert_array_equal(result1.true_labels, result2.true_labels)
        assert result1.dataset_info == result2.dataset_info
        assert result1.model_info == result2.model_info 