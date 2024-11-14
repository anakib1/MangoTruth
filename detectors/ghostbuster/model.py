import pathlib
from pickle import load
from typing import List

import numpy as np

from detectors.ghostbuster.features import extract_features
from detectors.interfaces import IDetector


class GhostbusterDetector(IDetector):
    def __init__(self, weights_folder: str):
        with open(pathlib.Path(weights_folder) / "clf.pkl", "rb") as f:
            self.clf = load(f)
        with open(pathlib.Path(weights_folder) / "unigram.pkl", "rb") as f:
            self.unigram = load(f)
        with open(pathlib.Path(weights_folder) / "trigram.pkl", "rb") as f:
            self.trigram = load(f)

    def predict_proba(self, text: str) -> (np.array, str):
        uni = self.unigram.predict_proba(text)
        tri = self.trigram.predict_proba(text)

        feats = extract_features([uni, tri])

        return self.clf.predict_proba(feats.reshape(1, -1)).flatten()

    def get_labels(self) -> List[str]:
        return ['Human', 'AI']
