import pathlib
from dataclasses import asdict
from os import getenv
from typing import Optional, Dict
from uuid import uuid4

import neptune
from neptune.types import File

from detectors.metrics import Conclusion, SplitConclusion
from detectors.nexus import Nexus, TrainingNexus


class NeptuneNexus(Nexus, TrainingNexus):

    def __init__(self, neptune_token: Optional[str] = None, neptune_proj: Optional[str] = 'mango/mango-truth'):
        if neptune_token is None:
            neptune_token = getenv('NEPTUNE_API_KEY')
        self.neptune_token = neptune_token
        self.neptune_proj = neptune_proj
        pathlib.Path('./cache').mkdir(parents=True, exist_ok=True)

        self.proj = neptune.init_project(neptune_proj, api_token=self.neptune_token, mode='sync')

    def get_run_id(self, run_id: uuid4):
        runs = self.proj.fetch_runs_table(query=f'`sys/name`:string CONTAINS "{str(run_id)}"').to_pandas()

        if len(runs) != 1:
            return None

        return runs['sys/id'][0]

    def get_run(self, run_id: uuid4):
        run_id = self.get_run_id(run_id)
        if run_id is not None:
            return neptune.init_run(with_id=run_id, project=self.neptune_proj, api_token=self.neptune_token,
                                    mode='sync')
        return None

    def get_or_create_run(self, run_id: uuid4):
        existing = self.get_run(run_id)
        if existing is None:
            return neptune.init_run(name=str(run_id), project=self.neptune_proj, api_token=self.neptune_token,
                                    mode='sync')
        return existing

    def store_run_weights(self, run_id: uuid4, content: bytes):
        run = self.get_or_create_run(run_id)
        run['checkpoint'].upload(File.from_content(content, extension='.pkl'), wait=True)

    def load_run_weights(self, run_id: uuid4) -> bytes:
        run = self.get_run(run_id)
        if run is None:
            raise Exception(f"Run for name {run_id} was not found.")
        pth = pathlib.Path('./cache').joinpath(f'weights-run-{run_id}.pkl')
        run['checkpoint'].download(str(pth))
        return open(str(pth), 'rb').read()

    def conclude_run(self, run_id: uuid4, conclusion: Conclusion, extra_data: Optional[Dict] = None):
        run = self.get_or_create_run(run_id)

        for split, concl in zip(['train', 'validation'],
                                [conclusion.train_conclusion, conclusion.validation_conclusion]):
            concl: SplitConclusion

            for k, v in asdict(concl.metrics).items():
                run[f'metrics/{split}/{k}'] = v

            for k, v in asdict(concl.representations).items():
                run[f'charts/{split}/{k}'].upload(v, wait=True)

        run[f'sys/tags'].add(conclusion.detector_handle)
        run[f'sys/tags'].add(conclusion.datasets)

        run.wait()

        self.store_run_weights(run_id, conclusion.weights)

        if extra_data is not None:
            for key, value in extra_data.items():
                run[key] = value

        run.wait()
