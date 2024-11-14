import numpy as np
from typing import List


class IDetector:
    def predict_proba(self, text: str) -> (np.array, str):  # predictions and explanation
        pass

    def get_labels(self) -> List[str]:
        pass

    def get_detector_name(self) -> str:
        return self.__class__.__name__
