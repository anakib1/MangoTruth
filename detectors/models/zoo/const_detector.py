from typing import List
import numpy as np
from detectors.interfaces import IDetector


class ConstDetector(IDetector):
    """Detector that returns constant predictions for testing"""
    def __init__(self, human_prob: float = 0.8):
        """
        Args:
            human_prob: Probability of predicting human class (default: 0.8)
        """
        self.labels = ["human", "ai"]
        self.human_prob = human_prob
        self.predictions = np.array([[human_prob, 1 - human_prob]])
    
    def get_labels(self) -> List[str]:
        return self.labels
    
    def predict_proba(self, text: str) -> np.ndarray:
        return self.predictions[0]
    
    def batch_predict(self, texts: List[str]) -> np.ndarray:
        return np.array([self.predictions[0]] * len(texts)) 