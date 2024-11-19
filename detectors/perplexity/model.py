from typing import List
import pickle

from detectors.utils.loading import ensure_type, ensure_obj

from detectors.interfaces import IDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm

NUM_SAMPLES = 1000
BATCH_SIZE = 8

NUM_SAMPLES = NUM_SAMPLES // BATCH_SIZE * BATCH_SIZE


def calculate_perplexity(dataset, text_column, num_samples):
    texts = []
    for i, sample in tqdm(enumerate(dataset)):
        if i >= num_samples:
            break
        texts.append(sample[text_column][:256])

    results = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        inputs = tokenizer(texts[i:i + BATCH_SIZE], padding='max_length', truncation=True, return_tensors='pt',
                           max_length=256)
        with torch.no_grad():
            logits = model(**inputs.to('cuda'))
        raw_perplexity = torch.nn.functional.cross_entropy(logits['logits'].permute(0, 2, 1), inputs['input_ids'],
                                                           reduction='none')

        raw_perplexity[inputs['input_ids'] == tokenizer.pad_token_id] = 0
        perplexities = raw_perplexity.sum(dim=1) / inputs['attention_mask'].sum(dim=1)
        results.append(perplexities.exp().cpu())

    return torch.concatenate(results, dim=0)


class PerplexityModel(IDetector):
    def __init__(self, model_handle: str):
        self.human_perplexity = None
        self.llm_perplexity = None
        self.model_handle = model_handle

        self.model = AutoModelForCausalLM.from_pretrained(model_handle, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_handle)

    def predict_proba(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs.to(self.model.device))

        perplexity = torch.nn.functional.cross_entropy(logits['logits'].permute(0, 2, 1), inputs['input_ids'],
                                                       reduction='mean').exp().cpu()

        if (perplexity)

    def get_labels(self) -> List[str]:
        return ['Human', 'LLM']

    def load_weights(self, weights: bytes) -> None:
        weights = pickle.loads(weights)
        ensure_type(weights, dict)
        ensure_obj(weights, 'human_perplexity')
        ensure_obj(weights, 'llm_perplexity')

        self.human_perplexity = weights['human_perplexity']
        self.llm_perplexity = weights['llm_perplexity']

    def store_weights(self) -> bytes:
        return pickle.dumps({
            'human_perplexity': self.human_perplexity,
            'llm_perplexity': self.llm_perplexity,
            'model_handle': self.model_handle})
