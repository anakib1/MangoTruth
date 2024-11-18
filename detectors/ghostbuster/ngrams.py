from collections import defaultdict

import numpy as np
from typing import List
from nltk import FreqDist, bigrams, trigrams
from transformers import AutoTokenizer
from detectors.interfaces import EstimationLanguageModel


class TrainableLanguageModel(EstimationLanguageModel):
    def __init__(self, tokenizer_handle="google/gemma-2-27b-it", discount=0.9):
        self.discount = discount
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_handle)

    def train(self, corpus_text: List[str]):
        raise NotImplementedError("This method should be implemented by subclasses.")


MIN_PROBABILITY = 1e-9

class UnigramModel(TrainableLanguageModel):
    def train(self, corpus_text: str):
        tokens_seq = self.tokenizer(corpus_text)['input_ids']
        tokens = []
        for token in tokens_seq:
            tokens.extend(token)
        self.total_tokens = len(tokens)
        self.unigram_freq = FreqDist(tokens)
        self.unigram_probabilities = {token: count / self.total_tokens for token, count in self.unigram_freq.items()}

    def get_text_log_proba(self, text):
        tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
        return tokens, np.array(
            [np.log(self.unigram_probabilities.get(token, MIN_PROBABILITY)) for token in tokens])


class TrigramModel(TrainableLanguageModel):
    def train(self, corpus_text: str):
        tokens_seq = self.tokenizer(corpus_text)['input_ids']
        tokens = []
        for token in tokens_seq:
            tokens.extend(token)

        self.unigram_freq = FreqDist(tokens)
        self.bigram_freq = FreqDist(bigrams(tokens))
        self.trigram_freq = FreqDist(trigrams(tokens))

        self.bigram_continuation_counts = defaultdict(int)
        self.trigram_continuation_counts = defaultdict(int)
        for (w1, w2) in self.bigram_freq:
            self.bigram_continuation_counts[w2] += 1
        for (w1, w2, w3) in self.trigram_freq:
            self.trigram_continuation_counts[(w2, w3)] += 1

        self.total_trigrams = sum(self.trigram_freq.values())

    def get_text_log_proba(self, text):
        tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
        trigram_probabilities = []

        for w1, w2, w3 in trigrams(tokens + [self.tokenizer.pad_token, self.tokenizer.pad_token]):
            trigram_count = self.trigram_freq.get((w1, w2, w3), 0)
            trigram_discounted = max(trigram_count - self.discount, 0) / self.total_trigrams
            bigram_continuation = (self.discount * self.trigram_continuation_counts.get((w2, w3),
                                                                                        0)) / self.total_trigrams
            bigram_count = self.bigram_freq.get((w2, w3), 0)
            bigram_prob = (max(bigram_count - self.discount, 0) / len(tokens) +
                           (self.discount * self.bigram_continuation_counts.get(w3, 0)) / len(tokens))
            trigram_prob = trigram_discounted + bigram_continuation * bigram_prob
            trigram_probabilities.append(trigram_prob)

        trigram_probabilities = np.array(trigram_probabilities)
        return tokens, np.log(trigram_probabilities, out=np.ones(len(tokens)) * np.log(MIN_PROBABILITY),
                              where=(trigram_probabilities != 0.0))
