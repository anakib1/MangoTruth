from typing import List
import pickle

from detectors.utils.loading import ensure_type, ensure_obj

from detectors.interfaces import IDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import numpy as np
import traceback
from detectors.utils.math import safe_sigmoid


class PerplexityModel(IDetector):
    def __init__(self, model_handle: str = None, perplexity_threshold: float = None, scaling_factor: float = None):
        self.model_handle = None
        self.model = None
        self.tokenizer = None
        self.openai = None
        self.perplexity_threshold = perplexity_threshold
        self.scaling_factor = scaling_factor
        self.set_model(model_handle)

    def set_model(self, model_handle):
        self.model_handle = model_handle
        if self.model_handle is not None:
            if '/' in model_handle:
                self.model = AutoModelForCausalLM.from_pretrained(model_handle, device_map='auto',
                                                                  torch_dtype='float16')
                self.tokenizer = AutoTokenizer.from_pretrained(model_handle)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.openai = OpenAI()

    def predict_openai(self, text):
        response = self.openai.completions.create(
            model=self.model_handle,
            prompt="<|endoftext|>" + text,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        logprobs = np.array(response.choices[0].logprobs.token_logprobs[1:])
        return np.exp(-np.mean(logprobs))

    def predict_llm(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs.to(self.model.device))

        return torch.nn.functional.cross_entropy(logits['logits'].permute(0, 2, 1), inputs['input_ids'],
                                                 reduction='mean').exp().cpu().numpy()

    def predict_proba(self, text):
        if self.model is not None:
            perplexity = self.predict_llm(text)
        else:
            perplexity = self.predict_openai(text)

        return safe_sigmoid(self.scaling_factor * (perplexity - self.perplexity_threshold))

    def get_labels(self) -> List[str]:
        return ['LLM', 'Human']

    def load_weights(self, weights: bytes) -> None:
        weights = pickle.loads(weights)
        ensure_type(weights, dict)
        self.perplexity_threshold = ensure_obj(weights, 'perplexity_threshold')
        self.model_handle = ensure_obj(weights, 'model_handle')
        self.scaling_factor = ensure_obj(weights, 'scaling_factor')

    def store_weights(self) -> bytes:
        return pickle.dumps({
            'perplexity_threshold': self.perplexity_threshold,
            'scaling_factor': self.scaling_factor,
            'model_handle': self.model_handle})
