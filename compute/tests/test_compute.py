import json
from dataclasses import asdict

import pytest

from compute.core.engine import ComputeEngine
from compute.core.mock_broker import MockMessageBroker
from compute.models.communication import ComputeRequest
from detectors.mocks import MockDetector
from compute.core.detectors import MockDetectorsEngine


@pytest.fixture
def setup_engine():
    mock_detector = MockDetector(labels=["label1", "label2"])

    mock_broker = MockMessageBroker()

    detectors = MockDetectorsEngine(mock_detector)
    engine = ComputeEngine(detectors_engine=detectors, broker=mock_broker)
    return engine, detectors, mock_detector


def test_bad_detector_request(setup_engine):
    engine, _, _ = setup_engine

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
    assert response.status == "FAILED"


def test_broker_on_request(setup_engine):
    engine, _, _ = setup_engine
    request = ComputeRequest(
        request_id="test_request_id",
        content="test_content",
        detector_name="mock_detector_non_existant"
    )
    json_request = json.dumps(asdict(request)).encode('utf-8')
    engine.broker.on_request(None, None, None, json_request)


def test_process_request(setup_engine):
    engine, _, _ = setup_engine

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
    assert response.status == "SUCCESS"


def test_get_detector_by_name(setup_engine):
    engine, detectors, mock_detector = setup_engine

    # Check if the detector is returned correctly
    detector = detectors.get_detector_by_name("mock_detector")
    assert detector == mock_detector

    # Check if None is returned when the detector is not found
    detector = detectors.get_detector_by_name("non_existent_detector")
    assert detector is None


def test_start_and_stop_consuming(setup_engine):
    engine, _, _ = setup_engine
    engine.start_consuming()
    engine.stop_consuming()


def test_close(setup_engine):
    engine, _, _ = setup_engine
    engine.close()
