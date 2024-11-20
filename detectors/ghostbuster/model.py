import pickle
from typing import List

import numpy as np
from sklearn.pipeline import Pipeline

from detectors.ghostbuster.features import extract_features
from detectors.interfaces import IDetector, EstimationLanguageModel
from detectors.utils.loading import ensure_type, ensure_obj


class GhostbusterDetector(IDetector):
    def __init__(self, clf: Pipeline = None, estimators: List[EstimationLanguageModel] = None):
        self.clf: Pipeline = clf
        self.estimators: List[EstimationLanguageModel] = estimators

    def predict_proba(self, text: str) -> np.array:
        feats = extract_features([x.get_text_log_proba(text)[1] for x in self.estimators])
        return self.clf.predict_proba(feats.reshape(1, -1)).flatten()

    def get_labels(self) -> List[str]:
        return ['Human', 'AI']

    def load_weights(self, weights: bytes) -> None:
        try:
            dct = pickle.loads(weights)
            ensure_type(dct, dict)

            self.clf = ensure_obj(dct, 'clf')
            self.estimators = ensure_obj(dct, 'estimator')

            ensure_type(self.clf, Pipeline)
            ensure_type(self.estimators, List)
            for estimator in self.estimators:
                ensure_type(estimator, EstimationLanguageModel)

        except Exception as e:
            raise Exception("Error occurred while loading weights.", e)

    def store_weights(self) -> bytes:
        return pickle.dumps({"clf": self.clf, "estimator": self.estimators})
