import numpy as np

from detectors.interfaces import IDetector
from detectors.detect_gpt.perturbator import Perturbator
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch


def batch_calculate_perplexity(texts, tokenizer, llm, batch_size):
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        inputs = tokenizer(texts[i:i + batch_size], padding='max_length', truncation=True,
                           return_tensors='pt',
                           max_length=tokenizer.model_max_length)
        with torch.no_grad():
            logits = llm(**inputs.to(llm.device))
        raw_perplexity = torch.nn.functional.cross_entropy(logits['logits'].permute(0, 2, 1), inputs['input_ids'],
                                                           reduction='none')

        raw_perplexity[inputs['input_ids'] == tokenizer.pad_token_id] = 0
        perplexities = raw_perplexity.sum(dim=1) / inputs['attention_mask'].sum(dim=1)
        results.append(perplexities.cpu())

    return torch.concatenate(results, dim=0).numpy()


class DetectGpt(IDetector):
    def __init__(self, perturbator_handle, perplexity_handle, num_perturbations=5):
        self.perturbator = Perturbator(perturbator_handle)
        self.perplexity_handle = perplexity_handle
        if '/' not in perplexity_handle:
            self.openai = OpenAI()
        else:
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_handle, device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(perplexity_handle)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_perturbations = num_perturbations

    def predict_openai(self, text):
        response = self.openai.completions.create(
            model=self.perplexity_handle,
            prompt="<|endoftext|>" + text,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        logprobs = np.array(response.choices[0].logprobs.token_logprobs[1:])
        return np.mean(logprobs)

    def predict_perplexities(self, texts: List[str]):
        if self.perplexity_model is not None:
            return batch_calculate_perplexity(texts, self.tokenizer, self.perplexity_model, 4)
        else:
            return np.array([self.predict_openai(perturbation) for perturbation in
                             tqdm(texts, desc="Calculating perplexities")])

    def predict_proba(self, text: str) -> np.array:
        perturbations = self.perturbator.perturbate([text] * self.num_perturbations)

        probas = self.predict_perplexities(perturbations)
        nu = np.mean(probas)
        d = self.predict_perplexities([text])[0] - nu
        var = np.sum((probas - nu) ** 2) / (self.num_perturbations - 1)

        if d / var > 0.005:
            return [0.0, 1.0]
        else:
            return [1.0, 0.0]

    def return_difference(self, text: str) -> np.array:
        perturbations = self.perturbator.perturbate([text] * self.num_perturbations)

        probas = self.predict_perplexities(perturbations)
        nu = np.mean(probas)
        d = self.predict_perplexities([text])[0] - nu
        var = np.sum((probas - nu) ** 2) / (self.num_perturbations - 1)

        return d / var

    def get_labels(self):
        return ['Human', 'AI']
