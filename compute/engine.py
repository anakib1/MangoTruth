from typing import List

from compute.interfaces import IMessageBroker
from compute.models.communication import ComputeRequest, ComputeResponse
from detectors.interfaces import IDetector
from compute.detectors import DetectorsEngine


class ComputeEngine:
    def __init__(self, detectors_engine: DetectorsEngine, broker: IMessageBroker):
        self.detectors_engine = detectors_engine
        self.broker = broker
        self.broker.set_process_request_method(self.process_request)

    def start_consuming(self):
        self.broker.start_consuming()

    def stop_consuming(self):
        self.broker.stop_consuming()

    def process_request(self, request: ComputeRequest) -> ComputeResponse:
        detector = self.detectors_engine.get_detector(request.detector_name)
        if detector is None:
            return ComputeResponse(
                status="FAILED",
                verdict=None,
                request_id=str(request.request_id)
            )
        predictions, status = detector.predict_proba(request.content)
        predictions_mapping = {"labels": [{"label": label, "probability": score} for label, score in
                                          zip(detector.get_labels(), predictions)]}

        return ComputeResponse(
            status=status,
            verdict=predictions_mapping,
            request_id=str(request.request_id)
        )

    def close(self):
        self.broker.close()
