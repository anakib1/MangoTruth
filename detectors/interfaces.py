from typing import Optional, Dict, Union, List, Tuple
from uuid import uuid4

from detectors.metrics import Conclusion
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




class Nexus:
    def load_run_weights(self, run_id: uuid4) -> bytes:
        """
        Loads model weights corresponding to run_id from remote nexus into bytes. Blocks until action is completed.
        Data will be cached in ./cache directory.
        :param run_id: UUID of the run
        :return: bytes loaded
        :throws: Exception in case run is unknown for the remote or other inconsistency
        """
        pass

    def store_run_weights(self, run_id: uuid4, content: bytes):
        """
        Stores content corresponding to run run_id to the remote nexus. Blocks until action is completed.
        :param run_id: UUID of the run
        :param content: bytes for model weights.
        :throws: Exception in case of remote server failure.
        """
        pass


class TrainingNexus:
    def conclude_run(self, run_id: uuid4, conclusion: Conclusion,
                     extra_data: Optional[Dict[str, Union[float, List[float]]]]):
        """
        Stores run conclusion in remote nexus.
        :param run_id: unique identifier of this run
        :param conclusion: object representing at least all the required metrics.
        :param extra_data: dictionary representing anything else that needs to be persisted. It will be persisted in
        folder-like structure
        :return:
        """
        pass


class CompletionLanguageModel:
    def complete_text(self, prefix: str, predict_log_proba: bool, n_words: int = None) -> Union[str, Tuple[List[str], np.array]]:
        """
        Generates the remaining suffix for the given beginning of some text.
        :param prefix: the beginning of some text
        :param predict_log_proba: if model should return log probabilities of underlying text
        :param n_words: number of words of the completed text
        :return: prefix + suffix split into tokens, probabilities of corresponding tokens.
        """
        pass

    def complete_texts(self, prefixes: List[str], predict_log_proba: bool, n_words: List[int] = None) -> List[Union[str, Tuple[List[str], np.array]]]:
        """
        The same as complete_text function, but generates the remaining suffix for K texts.
        """

class EstimationLanguageModel:
    def get_text_log_proba(self, text: str) -> Tuple[List[str], np.array]:
        """
        Returns log token probabilities for the given text.
        :param text: input text
        :return: Tuple of text split in model's tokens and np.array representing probabilities of corresponding tokens.
        """
        pass

