from detectors.interfaces import IDetector
from compute.models.communication import ComputeRequest, ComputeResponse
from typing import List


class ComputeEngine:
    # source - kafka that will provide requests
    # sink - kafka that will accept responses.
    def __init__(self, source, sink, detectors: List[IDetector]):
        self.source = source
        self.sink = sink
        self.detectors = detectors

    def do_work(self):
        if not self.source.is_empty():
            request: ComputeRequest = self.source.take()
            # Verify request, pass it to corresponding detector
            detector = self.detectors[0]
            resp = detector.predict_proba(request.content)
            self.sink.put(
                ComputeResponse("Missing", {x: y for x, y in zip(detector.get_labels(), resp)}, request.request_id))

    def has_work(self):
        return not self.source.is_empty()
