import numpy as np
import pytest

from detectors.mocks import MockDetector  # Assuming MockDetector is saved in a file named `mock_detector.py`


@pytest.fixture
def detector():
    detector_name = "test"
    return MockDetector()


def test_predict_proba(detector):
    # Arrange
    text = "This is a test sentence."

    # Act
    probabilities = detector.predict_proba(text)

    # Assert
    assert isinstance(probabilities, np.ndarray), "Expected predict_proba to return a NumPy array"
    assert probabilities.shape == (
        len(detector.get_labels()),), f"Expected array of shape ({len(detector.get_labels())},)"
    assert np.isclose(probabilities.sum(), 1), "Expected probabilities to sum to 1"


def test_get_labels(detector):
    # Act
    labels = detector.get_labels()

    # Assert
    assert isinstance(labels, list), "Expected get_labels to return a list"
    assert all(isinstance(label, str) for label in labels), "Expected all labels to be strings"
    assert len(labels) == len(detector.labels), "Expected the length of labels to match the internal labels"


def test_custom_labels():
    # Arrange
    custom_labels = ['Spam', 'Not Spam']
    detector_name = "test"
    detector = MockDetector(labels=custom_labels)

    # Act
    labels = detector.get_labels()

    # Assert
    assert labels == custom_labels, "Expected custom labels to be returned"
