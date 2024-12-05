"""
a lot of functions from here were copied from this repo
https://github.com/Xianjun-Yang/DNA-GPT/tree/main
"""
import re
from typing import List
from collections import Counter
from detectors.dna.config import NGramsConfig


class NGramsProcessor:
    def __init__(self, config: NGramsConfig):
        self.config = config

    def tokenize(self, text: str):
        """Tokenize input text into a list of tokens.
        This approach aims to replicate the approach taken by Chin-Yew Lin in
        the original ROUGE implementation.
        Args:
        text: A text blob to tokenize.
        stemmer: An optional stemmer.
        Returns:
        A list of string tokens extracted from input text.
        """
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        tokens = re.split(r"\s+", text)
        tokens = [self.config.stemmer.stem(x) if len(x) > 3 else x for x in tokens if x not in self.config.stopwords]
        tokens = [x for x in tokens if re.match(r"^\w+$", x)]
        return tokens

    @staticmethod
    def create_ngrams(tokens: List[str], n: int) -> Counter:
        """Creates ngrams from the given list of tokens.
        Args:
          tokens: A list of tokens from which ngrams are created.
          n: Number of tokens to use, e.g. 2 for bigrams.
        Returns:
          A dictionary mapping each ngram to the number of occurrences.
        """
        ngrams = Counter()
        for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    @staticmethod
    def calculate_intersection(ngrams1: Counter, ngrams2: Counter) -> Counter:
        """
        return the intersection of ngrams1 and ngrams2.
        """
        ngrams3 = Counter()
        for key in ngrams1.keys():
            ngrams3[key] = min(ngrams1[key], ngrams2[key])
        return ngrams3

    @staticmethod
    def get_ngrams_number(ngrams: Counter) -> int:
        return sum(ngrams.values())

    def calculate_metric(self, X: str, Y: List[str]) -> float:
        K = len(Y)
        elems = []
        ngrams_X = [self.create_ngrams(self.tokenize(X), N_i) for N_i in range(self.config.N_min, self.config.N_max + 1)]
        for k in range(K):
            length_k = len(self.tokenize(Y[k]))
            ngrams_Y = [self.create_ngrams(self.tokenize(Y[k]), N_i) for N_i in range(self.config.N_min, self.config.N_max + 1)]
            s = 0
            for N_i in range(self.config.N_min, self.config.N_max + 1):
                s += self.config.func(N_i) * self.get_ngrams_number(self.calculate_intersection(ngrams_X[N_i - self.config.N_min], ngrams_Y[N_i - self.config.N_min])) \
                     / length_k / max(self.get_ngrams_number(ngrams_X[N_i - self.config.N_min]), 1)
            elems.append(s)
        return sum(elems) / K
