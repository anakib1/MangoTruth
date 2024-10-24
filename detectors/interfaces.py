import numpy as np
from typing import List


class IDetector:
    def predict_proba(self, text: str) -> np.array:
        pass

    def get_labels(self) -> List[str]:
        pass
