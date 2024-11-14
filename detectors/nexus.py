from typing import Optional, Dict, Union, List
from uuid import uuid4

from detectors.metrics import Conclusion


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
