import unittest
from compute.models.communication import ComputeRequest
from compute.engine import ComputeEngine
from compute.rabbitmq_broker import RabbitMQBroker
from detectors.mocks import MockDetector


class TestComputeEngine(unittest.TestCase):
    def setUp(self):
        self.mock_detector = MockDetector(detector_name="mock_detector", labels=["label1", "label2"])

        self.mock_broker = RabbitMQBroker(
            source_queue_name="requests_queue",
            response_queue_name="responses_queue",
            rabbitmq_host="localhost",
            rabbitmq_port=5672

        )

        self.engine = ComputeEngine(detectors=[self.mock_detector], default_detector=self.mock_detector,
                                    broker=self.mock_broker)

    def test_bad_detector_request(self):
        # Create a mock request
        request = ComputeRequest(
            request_id="test_request_id",
            content="test_content",
            detector_name="mock_detector_non_existant"
        )

        # Call process_request
        response = self.engine.process_request(request)

        # Verify the response structure
        self.assertEqual(response.request_id, "test_request_id")
        self.assertEqual(response.explanation, "Prediction based on provided content")

    def test_process_request(self):
        # Create a mock request
        request = ComputeRequest(
            request_id="test_request_id",
            content="test_content",
            detector_name="mock_detector"
        )

        # Call process_request
        response = self.engine.process_request(request)

        # Verify the response structure
        self.assertEqual(response.request_id, "test_request_id")
        self.assertEqual(response.explanation, "Prediction based on provided content")

    def test_get_detector_by_name(self):
        # Check if the detector is returned correctly
        detector = self.engine.get_detector_by_name("mock_detector")
        self.assertEqual(detector, self.mock_detector)

        # Check if None is returned when the detector is not found
        detector = self.engine.get_detector_by_name("non_existent_detector")
        self.assertEqual(detector, self.engine.default_detector)

    def test_start_and_stop_consuming(self):
        self.engine.start_consuming()
        self.engine.stop_consuming()

    def test_close(self):
        self.engine.close()


if __name__ == "__main__":
    unittest.main()
