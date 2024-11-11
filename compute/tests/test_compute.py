import pytest

from compute.engine import ComputeEngine
from compute.mock_broker import LocalMessageBroker
from compute.models.communication import ComputeRequest
from detectors.mocks import MockDetector


@pytest.fixture
def setup_engine():
    mock_detector = MockDetector(detector_name="mock_detector", labels=["label1", "label2"])

    mock_broker = LocalMessageBroker()

    engine = ComputeEngine(detectors=[mock_detector], default_detector=mock_detector, broker=mock_broker)
    return engine, mock_detector


def test_bad_detector_request(setup_engine):
    engine, _ = setup_engine

    # Create a mock request
    request = ComputeRequest(
        request_id="test_request_id",
        content="test_content",
        detector_name="mock_detector_non_existant"
    )

    # Call process_request
    response = engine.process_request(request)

    # Verify the response structure
    assert response.request_id == "test_request_id"
    assert response.explanation == "Prediction based on provided content"


def test_process_request(setup_engine):
    engine, _ = setup_engine

    # Create a mock request
    request = ComputeRequest(
        request_id="test_request_id",
        content="test_content",
        detector_name="mock_detector"
    )

    # Call process_request
    response = engine.process_request(request)

    # Verify the response structure
    assert response.request_id == "test_request_id"
    assert response.explanation == "Prediction based on provided content"


def test_get_detector_by_name(setup_engine):
    engine, mock_detector = setup_engine

    # Check if the detector is returned correctly
    detector = engine.get_detector_by_name("mock_detector")
    assert detector == mock_detector

    # Check if None is returned when the detector is not found
    detector = engine.get_detector_by_name("non_existent_detector")
    assert detector == engine.default_detector


def test_start_and_stop_consuming(setup_engine):
    engine, _ = setup_engine
    engine.start_consuming()
    engine.stop_consuming()


def test_close(setup_engine):
    engine, _ = setup_engine
    engine.close()
