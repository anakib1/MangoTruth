from typing import List
from models.completion import OpenAICompletionModel
from .config import BlackBoxDnaConfig
from .ngrams import NGramsProcessor
from interfaces import IDetector
import numpy as np

BLACK_BOX_DNA_BYTEARRAY = b"BlackBoxDNADetector"


class BlackBoxDNADetector(IDetector):
    def __init__(self, config: BlackBoxDnaConfig):
        self.config = config
        self.model = OpenAICompletionModel(self.config.model_config)
        self.ngrams_processor = NGramsProcessor(self.config.ngrams_config)

    def get_labels(self) -> List[str]:
        return ["Human", "AI"]

    def predict_proba(self, text: str) -> np.array:
        words = text.split()
        divider_num = int(len(words) * (1 - self.config.truncation))
        beginning, X_ending = " ".join(words[:divider_num]), " ".join(words[divider_num:])
        Y_endings = self.model.complete_texts([beginning] * self.config.K, False)
        metric = self.ngrams_processor.calculate_metric(X_ending, Y_endings)
        if self.config.threshold > metric:
            return np.array([0.0, 1.0])
        else:
            return np.array([1.0, 0.0])


    def store_weights(self) -> bytes:
        return BLACK_BOX_DNA_BYTEARRAY

    def load_weights(self, weights: bytes):
        if weights != BLACK_BOX_DNA_BYTEARRAY:
            raise TypeError("The weights are not from BlackBoxDNADetector")
