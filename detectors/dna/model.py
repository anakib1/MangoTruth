from typing import List
from detectors.models.completion import OpenAICompletionModel
from detectors.dna.config import BlackBoxDnaConfig
from detectors.dna.ngrams import NGramsProcessor
from detectors.interfaces import IDetector
from detectors.dna.serialise import config_to_bytes, config_from_bytes
import numpy as np

BLACK_BOX_DNA_BYTEARRAY = b"BlackBoxDNADetector"


class BlackBoxDNADetector(IDetector):
    def __init__(self, config: BlackBoxDnaConfig = None):
        if config:
            self._load_from_config(config)

    def _load_from_config(self, config: BlackBoxDnaConfig):
        self.config = config
        self.model = OpenAICompletionModel(self.config.model_config)
        self.ngrams_processor = NGramsProcessor(self.config.ngrams_config)

    def get_labels(self) -> List[str]:
        return ["Human", "AI"]

    def calculate_bscore(self, text: str):
        words = text.split()
        divider_num = int(len(words) * (1 - self.config.truncation))
        beginning, X_ending = " ".join(words[:divider_num]), " ".join(words[divider_num:])
        Y_endings = self.model.complete_texts([beginning] * self.config.K, False, n_words=[len(words)] * self.config.K)
        metric = self.ngrams_processor.calculate_metric(X_ending, Y_endings)
        return metric

    def predict_proba(self, text: str) -> np.array:
        metric = self.calculate_bscore(text)
        if metric > self.config.threshold:
            return np.array([0.0, 1.0])
        else:
            return np.array([1.0, 0.0])

    def store_weights(self) -> bytes:
        return config_to_bytes(self.config)

    def load_weights(self, weights: bytes):
        config = config_from_bytes(weights)
        if not isinstance(config, BlackBoxDnaConfig):
            raise TypeError("The config should be BlackBoxDnaConfig")
        self._load_from_config(config)
