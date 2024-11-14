import pickle
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from detectors.ghostbuster.features import extract_features
from detectors.ghostbuster.ngrams import UnigramModel, TrigramModel
from detectors.interfaces import IDetector
from detectors.utils.loading import ensure_type, ensure_obj


class GhostbusterDetector(IDetector):

    def __init__(self):
        self.clf: LogisticRegressionCV = None
        self.trigram: TrigramModel = None
        self.unigram: UnigramModel = None

    def predict_proba(self, text: str) -> np.array:
        uni = self.unigram.predict_proba(text)
        tri = self.trigram.predict_proba(text)

        feats = extract_features([uni, tri])

        return self.clf.predict_proba(feats.reshape(1, -1)).flatten()

    def get_labels(self) -> List[str]:
        return ['Human', 'AI']

    def load_weights(self, weights: bytes) -> None:
        try:
            dct = pickle.loads(weights)
            ensure_type(dct, dict)

            self.clf = ensure_obj(dct, 'clf')
            self.unigram = ensure_obj(dct, 'unigram')
            self.trigram = ensure_obj(dct, 'trigram')

            ensure_type(self.clf, LogisticRegressionCV)
            ensure_type(self.unigram, UnigramModel)
            ensure_type(self.trigram, TrigramModel)

        except Exception as e:
            raise Exception("Error occurred while loading weights.", e)

    def store_weights(self) -> bytes:
        return pickle.dumps({"clf": self.clf, "unigram": self.unigram, "trigram": self.trigram})
