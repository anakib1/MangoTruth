from typing import List

import numpy as np


class IDetector:
    def predict_proba(self, text: str) -> np.array:
        """
        Returns probabilities in the same order as model's labels.
        :param text: input text sequence to classify
        :return: array of probabilities. Probabilities always some to one,
        """
        pass

    def get_labels(self) -> List[str]:
        """
        Returns detector's possible labels.
        :return: list of labels in readable form.
        """
        pass

    def store_weights(self) -> bytes:
        """
        Returns classifier's weights in bytes representation. The only sane assumption about the bytes should be,
        that method `load_weights` will successfully load full model state from the passed argument.
        :return: bytes of model's weights
        """
        pass

    def load_weights(self, weights: bytes) -> None:
        """
        Loads all classifiers weights from bytes content. This content should be created by
        `store_weights` method.
        :raises IllegalArgumentException in case if weights are malformed
        :param weights: bytes containing weights
        """
        pass
