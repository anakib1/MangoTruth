import os
from typing import Tuple, List

import numpy as np

from detectors.interfaces import EstimationLanguageModel
from openai import OpenAI


class OpenaiProbabilityEstimator(EstimationLanguageModel):
    def __init__(self, api_key: str = None, model_name: str = "davinci-002"):
        self.api_key = api_key
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY', api_key))
        self.model_name = model_name

    def get_text_log_proba(self, text: str) -> Tuple[List[str], np.array]:
        response = self.openai.completions.create(
            model=self.model_name,
            prompt="<|endoftext|>" + text,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        return response.choices[0].logprobs.tokens[1:], np.array(response.choices[0].logprobs.token_logprobs[1:])

    def __reduce__(self):
        return self.__class__, (self.api_key, self.model_name)
