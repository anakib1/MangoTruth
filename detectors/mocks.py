from typing import Optional, List

import numpy as np

from detectors.interfaces import IDetector


class MockDetector(IDetector):
    def __init__(self, detector_name: str, labels: Optional[List[str]] = None):
        self.detector_name = detector_name
        if labels is None:
            labels = ['AI', 'Human']
        self.labels = labels

    def predict_proba(self, text: str) -> (np.array, str):
        ret = np.random.random(size=(len(self.labels),))
        status = "SUCCESS"
        return ret / ret.sum(), status

    def get_labels(self) -> List[str]:
        return self.labels

    def get_detector_name(self) -> str:
        return self.detector_name
