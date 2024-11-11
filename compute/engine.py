from detectors.interfaces import IDetector
from compute.models.communication import ComputeRequest, ComputeResponse
from compute.interfaces import IMessageBroker
from typing import List


class ComputeEngine:
    def __init__(self, detectors: List[IDetector], default_detector: IDetector, broker: IMessageBroker):
        self.detectors = detectors
        self.default_detector = default_detector
        self.broker = broker
        self.broker.set_process_request_method(self.process_request)

    def start_consuming(self):
        self.broker.start_consuming()

    def stop_consuming(self):
        self.broker.stop_consuming()

    def process_request(self, request: ComputeRequest) -> ComputeResponse:
        detector = self.get_detector_by_name(request.detector_name)
        predictions, explanation = detector.predict_proba(request.content)
        predictions_mapping = {label: score for label, score in zip(detector.get_labels(), predictions)}

        return ComputeResponse(
            explanation=explanation,
            predictions=predictions_mapping,
            request_id=request.request_id
        )

    def get_detector_by_name(self, detector_name: str) -> IDetector:
        # Search for the detector by name or return default if not found
        for detector in self.detectors:
            if detector.get_detector_name() == detector_name:
                return detector

        return self.default_detector

    def close(self):
        self.broker.close()
