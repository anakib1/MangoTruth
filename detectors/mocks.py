import numpy as np
from typing import Optional, List

from models.interfaces import IDetector


class MockDetector(IDetector):
    def __init__(self, labels: Optional[List[str]] = None):
        if labels is None:
            labels = ['AI', 'Human']
        self.labels = labels

    def predict_proba(self, text: str) -> np.array:
        ret = np.random.random(size=(len(self.labels),))
        return ret / ret.sum()

    def get_labels(self) -> List[str]:
        return self.labels
