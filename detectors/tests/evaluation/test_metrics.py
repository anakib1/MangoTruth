"""Tests for centralized metrics calculation components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from detectors.evaluation.metrics import (
    MetricsCalculator,
    MultiClassMetricsCalculator,
    MetricsCalculatorFactory
)
from detectors.metrics import SplitConclusion, ClassificationMetrics


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        calculator = MetricsCalculator()
        
        assert calculator.threshold == 0.5
        assert calculator.include_visualizations is True
        assert calculator.fpr_thresholds == [0.01, 0.1]
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        calculator = MetricsCalculator(
            threshold=0.7,
            include_visualizations=False,
            fpr_thresholds=[0.05, 0.2]
        )
        
        assert calculator.threshold == 0.7
        assert calculator.include_visualizations is False
        assert calculator.fpr_thresholds == [0.05, 0.2]
    
    @patch('detectors.evaluation.metrics.calculate_classification')
    @patch('detectors.evaluation.metrics.safe_roc')
    @patch('detectors.evaluation.metrics.tpr_at_fpr_threshold')
    def test_calculate_metrics_basic(self, mock_tpr_at_fpr, mock_safe_roc, mock_calculate):
        """Test basic metrics calculation."""
        # Setup mocks
        mock_metrics = Mock()
        mock_calculate.return_value = mock_metrics
        mock_safe_roc.return_value = ([0.0, 0.1, 1.0], [0.0, 0.8, 1.0], 0.9)
        mock_tpr_at_fpr.return_value = 0.75
        
        calculator = MetricsCalculator()
        true_labels = np.array([0, 1, 0, 1])
        predictions = np.array([0.2, 0.8, 0.1, 0.9])
        
        result = calculator.calculate_metrics(true_labels, predictions)
        
        # Verify function calls
        mock_calculate.assert_called_once_with(true_labels, predictions)
        mock_safe_roc.assert_called_once_with(true_labels, predictions)
        
        # Should call tpr_at_fpr for each threshold
        assert mock_tpr_at_fpr.call_count == 2
        
        # Verify result structure
        assert isinstance(result, SplitConclusion)
        assert result.metrics == mock_metrics
        
        # Check custom TPR metrics were added
        assert hasattr(mock_metrics, 'tpr_at_1_percent_fpr')
        assert hasattr(mock_metrics, 'tpr_at_10_percent_fpr')
    
    @patch('detectors.evaluation.metrics.calculate_classification')
    @patch('detectors.evaluation.metrics.safe_roc')
    @patch('detectors.evaluation.metrics.draw_classification')
    def test_calculate_metrics_with_visualizations(self, mock_draw, mock_safe_roc, mock_calculate):
        """Test metrics calculation with visualizations."""
        # Setup mocks
        mock_metrics = Mock()
        mock_calculate.return_value = mock_metrics
        mock_safe_roc.return_value = ([0.0, 0.1, 1.0], [0.0, 0.8, 1.0], 0.9)
        mock_representations = Mock()
        mock_draw.return_value = mock_representations
        
        calculator = MetricsCalculator(include_visualizations=True)
        true_labels = np.array([0, 1, 0, 1])
        predictions = np.array([0.2, 0.8, 0.1, 0.9])
        
        result = calculator.calculate_metrics(true_labels, predictions)
        
        # Should generate visualizations
        mock_draw.assert_called_once_with(true_labels, predictions)
        assert result.representations == mock_representations
    
    @patch('detectors.evaluation.metrics.calculate_classification')
    @patch('detectors.evaluation.metrics.safe_roc')
    @patch('detectors.evaluation.metrics.draw_classification')
    def test_calculate_metrics_without_visualizations(self, mock_draw, mock_safe_roc, mock_calculate):
        """Test metrics calculation without visualizations."""
        mock_metrics = Mock()
        mock_calculate.return_value = mock_metrics
        mock_safe_roc.return_value = ([0.0, 0.1, 1.0], [0.0, 0.8, 1.0], 0.9)
        
        calculator = MetricsCalculator(include_visualizations=False)
        true_labels = np.array([0, 1, 0, 1])
        predictions = np.array([0.2, 0.8, 0.1, 0.9])
        
        result = calculator.calculate_metrics(true_labels, predictions)
        
        # Should not generate visualizations
        mock_draw.assert_not_called()
        assert result.representations is None
    
    @patch('detectors.evaluation.metrics.calculate_classification')
    @patch('detectors.evaluation.metrics.safe_roc')
    @patch('detectors.evaluation.metrics.draw_classification')
    def test_calculate_metrics_visualization_error(self, mock_draw, mock_safe_roc, mock_calculate):
        """Test handling of visualization errors."""
        mock_metrics = Mock()
        mock_calculate.return_value = mock_metrics
        mock_safe_roc.return_value = ([0.0, 0.1, 1.0], [0.0, 0.8, 1.0], 0.9)
        mock_draw.side_effect = Exception("Visualization failed")
        
        calculator = MetricsCalculator(include_visualizations=True)
        true_labels = np.array([0, 1, 0, 1])
        predictions = np.array([0.2, 0.8, 0.1, 0.9])
        
        result = calculator.calculate_metrics(true_labels, predictions)
        
        # Should handle error gracefully
        assert result.representations is not None
        assert result.representations.roc_curve is None
        assert result.representations.clf_report is None
    
    def test_calculate_metrics_with_custom_threshold(self):
        """Test metrics calculation with custom threshold."""
        calculator = MetricsCalculator()
        true_labels = np.array([0, 1, 0, 1])
        predictions = np.array([0.2, 0.8, 0.1, 0.9])
        
        with patch('detectors.evaluation.metrics.calculate_classification') as mock_calculate:
            mock_calculate.return_value = Mock()
            
            calculator.calculate_metrics(true_labels, predictions, threshold=0.6)
            
            # The threshold should be available for use (though calculate_classification might not use it)
            mock_calculate.assert_called_once_with(true_labels, predictions)
    
    def test_calculate_batch_metrics(self):
        """Test batch metrics calculation."""
        calculator = MetricsCalculator()
        
        results = {
            "dataset1": (np.array([0, 1]), np.array([0.2, 0.8])),
            "dataset2": (np.array([1, 0]), np.array([0.9, 0.1]))
        }
        
        with patch.object(calculator, 'calculate_metrics') as mock_calc:
            mock_conclusion1 = Mock()
            mock_conclusion2 = Mock()
            mock_calc.side_effect = [mock_conclusion1, mock_conclusion2]
            
            batch_results = calculator.calculate_batch_metrics(results, custom_param="test")
        
        # Should call calculate_metrics for each dataset
        assert mock_calc.call_count == 2
        
        # Check calls were made with correct parameters
        call_args_list = mock_calc.call_args_list
        assert np.array_equal(call_args_list[0][0][0], np.array([0, 1]))  # true_labels for dataset1
        assert np.array_equal(call_args_list[0][0][1], np.array([0.2, 0.8]))  # predictions for dataset1
        assert call_args_list[0][1]["custom_param"] == "test"  # kwargs passed through
        
        # Check results
        assert batch_results["dataset1"] == mock_conclusion1
        assert batch_results["dataset2"] == mock_conclusion2


class TestMultiClassMetricsCalculator:
    """Tests for MultiClassMetricsCalculator class."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        calculator = MultiClassMetricsCalculator()
        assert calculator.average_method == 'macro'
        assert calculator.include_visualizations is True
        
        # Custom initialization
        calculator = MultiClassMetricsCalculator(
            average_method='micro',
            include_visualizations=False
        )
        assert calculator.average_method == 'micro'
        assert calculator.include_visualizations is False
    
    @patch('detectors.evaluation.metrics.MetricsCalculator')
    def test_calculate_metrics_multiclass_conversion(self, mock_standard_calc_class):
        """Test multi-class to binary conversion."""
        mock_standard_calc = Mock()
        mock_conclusion = Mock()
        mock_standard_calc.calculate_metrics.return_value = mock_conclusion
        mock_standard_calc_class.return_value = mock_standard_calc
        
        calculator = MultiClassMetricsCalculator()
        
        # Multi-class predictions and labels
        true_labels = np.array([0, 1, 2, 0])
        predictions = np.array([
            [0.8, 0.1, 0.1],  # Predicted: 0, True: 0 ✓
            [0.2, 0.7, 0.1],  # Predicted: 1, True: 1 ✓
            [0.1, 0.1, 0.8],  # Predicted: 2, True: 2 ✓
            [0.3, 0.6, 0.1]   # Predicted: 1, True: 0 ✗
        ])
        
        result = calculator.calculate_metrics(true_labels, predictions)
        
        # Should create standard calculator
        mock_standard_calc_class.assert_called_once_with(include_visualizations=True)
        
        # Should call calculate_metrics on the standard calculator
        mock_standard_calc.calculate_metrics.assert_called_once()
        
        # Check the converted data passed to standard calculator
        call_args = mock_standard_calc.calculate_metrics.call_args[0]
        binary_true = call_args[0]
        binary_pred = call_args[1]
        
        # Should convert to binary correct/incorrect
        expected_binary_true = np.array([1, 1, 1, 0])  # First 3 correct, last incorrect
        np.testing.assert_array_equal(binary_true, expected_binary_true)
        
        assert result == mock_conclusion
    
    def test_calculate_metrics_1d_predictions(self):
        """Test with 1D predictions (already class predictions)."""
        calculator = MultiClassMetricsCalculator()
        
        true_labels = np.array([0, 1, 2, 0])
        predictions = np.array([0, 1, 2, 1])  # 1D array
        
        with patch('detectors.evaluation.metrics.MetricsCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc.calculate_metrics.return_value = Mock()
            mock_calc_class.return_value = mock_calc
            
            calculator.calculate_metrics(true_labels, predictions)
            
            # Should still work with 1D predictions
            mock_calc.calculate_metrics.assert_called_once()


class TestMetricsCalculatorFactory:
    """Tests for MetricsCalculatorFactory class."""
    
    def test_create_calculator_binary(self):
        """Test creating binary metrics calculator."""
        calculator = MetricsCalculatorFactory.create_calculator('binary')
        
        assert isinstance(calculator, MetricsCalculator)
        assert calculator.threshold == 0.5  # Default threshold
    
    def test_create_calculator_binary_with_kwargs(self):
        """Test creating binary calculator with custom parameters."""
        calculator = MetricsCalculatorFactory.create_calculator(
            'binary', 
            threshold=0.7, 
            include_visualizations=False
        )
        
        assert isinstance(calculator, MetricsCalculator)
        assert calculator.threshold == 0.7
        assert calculator.include_visualizations is False
    
    def test_create_calculator_multiclass(self):
        """Test creating multiclass metrics calculator."""
        calculator = MetricsCalculatorFactory.create_calculator('multiclass')
        
        assert isinstance(calculator, MultiClassMetricsCalculator)
        assert calculator.average_method == 'macro'  # Default
    
    def test_create_calculator_multiclass_with_kwargs(self):
        """Test creating multiclass calculator with custom parameters."""
        calculator = MetricsCalculatorFactory.create_calculator(
            'multiclass',
            average_method='micro',
            include_visualizations=False
        )
        
        assert isinstance(calculator, MultiClassMetricsCalculator)
        assert calculator.average_method == 'micro'
        assert calculator.include_visualizations is False
    
    def test_create_calculator_unsupported_type(self):
        """Test error handling for unsupported task types."""
        with pytest.raises(ValueError, match="Unsupported task type"):
            MetricsCalculatorFactory.create_calculator('unsupported')
    
    def test_create_for_model_binary(self):
        """Test creating calculator based on binary model."""
        mock_model = Mock()
        mock_model.get_labels.return_value = ["negative", "positive"]
        
        calculator = MetricsCalculatorFactory.create_for_model(mock_model)
        
        assert isinstance(calculator, MetricsCalculator)
    
    def test_create_for_model_multiclass(self):
        """Test creating calculator based on multi-class model."""
        mock_model = Mock()
        mock_model.get_labels.return_value = ["class1", "class2", "class3"]
        
        calculator = MetricsCalculatorFactory.create_for_model(mock_model)
        
        assert isinstance(calculator, MultiClassMetricsCalculator)
    
    def test_create_for_model_with_kwargs(self):
        """Test creating calculator for model with additional parameters."""
        mock_model = Mock()
        mock_model.get_labels.return_value = ["neg", "pos"]
        
        calculator = MetricsCalculatorFactory.create_for_model(
            mock_model,
            threshold=0.8,
            include_visualizations=False
        )
        
        assert isinstance(calculator, MetricsCalculator)
        assert calculator.threshold == 0.8
        assert calculator.include_visualizations is False
    
    def test_create_for_model_error_handling(self):
        """Test error handling when model doesn't have get_labels method."""
        mock_model = Mock()
        mock_model.get_labels.side_effect = AttributeError("No get_labels method")
        
        # Should fallback to binary with warning
        calculator = MetricsCalculatorFactory.create_for_model(mock_model)
        
        assert isinstance(calculator, MetricsCalculator)
    
    def test_create_for_model_no_get_labels(self):
        """Test creating calculator for model without get_labels method."""
        mock_model = Mock(spec=[])  # No get_labels method
        
        calculator = MetricsCalculatorFactory.create_for_model(mock_model)
        
        # Should default to binary
        assert isinstance(calculator, MetricsCalculator)


class TestMetricsIntegration:
    """Integration tests for metrics calculation."""
    
    def test_end_to_end_binary_metrics(self):
        """Test complete binary metrics calculation workflow."""
        from detectors.tests.conftest import create_binary_classification_data
        
        true_labels, predictions = create_binary_classification_data(100)
        
        calculator = MetricsCalculator(include_visualizations=False)
        result = calculator.calculate_metrics(true_labels, predictions)
        
        # Should return valid SplitConclusion
        assert isinstance(result, SplitConclusion)
        assert hasattr(result.metrics, 'accuracy')
        assert hasattr(result.metrics, 'f1')
        assert hasattr(result.metrics, 'auc')
        assert hasattr(result.metrics, 'precision')
        assert hasattr(result.metrics, 'recall')
        
        # Metrics should be in reasonable ranges
        assert 0 <= result.metrics.accuracy <= 1
        assert 0 <= result.metrics.f1 <= 1
        assert 0 <= result.metrics.auc <= 1
    
    def test_batch_metrics_integration(self):
        """Test batch metrics calculation integration."""
        from detectors.tests.conftest import create_binary_classification_data
        
        # Create multiple datasets
        results = {}
        for dataset_name in ["train", "val", "test"]:
            true_labels, predictions = create_binary_classification_data(50)
            results[dataset_name] = (true_labels, predictions)
        
        calculator = MetricsCalculator(include_visualizations=False)
        batch_results = calculator.calculate_batch_metrics(results)
        
        # Should have results for all datasets
        assert len(batch_results) == 3
        assert "train" in batch_results
        assert "val" in batch_results
        assert "test" in batch_results
        
        # All results should be SplitConclusion objects
        for result in batch_results.values():
            assert isinstance(result, SplitConclusion)
            assert hasattr(result.metrics, 'accuracy')
    
    def test_factory_integration_with_real_model(self):
        """Test factory integration with realistic model mock."""
        # Create a realistic model mock
        mock_model = Mock()
        mock_model.get_labels.return_value = ["human", "ai"]
        mock_model.__class__.__name__ = "HuggingFaceModel"
        
        # Create calculator using factory
        calculator = MetricsCalculatorFactory.create_for_model(
            mock_model,
            threshold=0.6,
            include_visualizations=True
        )
        
        assert isinstance(calculator, MetricsCalculator)
        assert calculator.threshold == 0.6
        assert calculator.include_visualizations is True
        
        # Test with sample data
        from detectors.tests.conftest import create_binary_classification_data
        true_labels, predictions = create_binary_classification_data(20)
        
        result = calculator.calculate_metrics(true_labels, predictions)
        assert isinstance(result, SplitConclusion) 