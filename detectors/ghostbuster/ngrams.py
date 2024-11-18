from collections import defaultdict

import numpy as np
from nltk import FreqDist, bigrams, trigrams
from transformers import AutoTokenizer


class LanguageModel:
    def __init__(self, tokenizer_handle="google/gemma-2-27b-it", discount=0.9):
        self.discount = discount
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_handle)

    def train(self, corpus_text):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict_proba(self, text):
        raise NotImplementedError("This method should be implemented by subclasses.")


class UnigramModel(LanguageModel):
    def train(self, corpus_text):
        tokens = self.tokenizer.tokenize(corpus_text)
        self.total_tokens = len(tokens)
        self.unigram_freq = FreqDist(tokens)
        self.unigram_probabilities = {token: count / self.total_tokens for token, count in self.unigram_freq.items()}

    def predict_proba(self, text):
        tokens = self.tokenizer.tokenize(text)
        return np.array([self.unigram_probabilities.get(token, 0.0) for token in tokens])


class TrigramModel(LanguageModel):
    def train(self, corpus_text):
        tokens = self.tokenizer.tokenize(corpus_text)

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

    def predict_proba(self, text):
        tokens = self.tokenizer.tokenize(text)
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

        return np.array(trigram_probabilities if trigram_probabilities else [0.0])


if __name__ == '__main__':
    corpus_text = "Your training corpus text here."

    unigram_model = UnigramModel()
    unigram_model.train(corpus_text)
    print("Unigram Probabilities:", unigram_model.predict_proba("Some text to evaluate"))

    trigram_model = TrigramModel()
    trigram_model.train(corpus_text)
    print("Trigram Probabilities:", trigram_model.predict_proba("Some text to evaluate"))
