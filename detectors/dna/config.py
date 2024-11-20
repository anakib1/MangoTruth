from dataclasses import dataclass
from nltk.stem import StemmerI
from nltk.stem.porter import PorterStemmer
from typing import Callable
import spacy
import math
from detectors.models.config import CompletionModelConfig


@dataclass
class NGramsConfig:
    N_min: int
    N_max: int
    stemmer: StemmerI
    func: Callable[[int], float]
    stopwords: list[str]


class NGramsConfigGenerator(NGramsConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__(
                N_min=4,
                N_max=25,
                stemmer=PorterStemmer(),
                func=lambda n: n * math.log(n),
                stopwords=spacy.load("en_core_web_sm").Defaults.stop_words
            )
        else:
            raise NotImplementedError("There are no such parameter set name")


@dataclass
class BlackBoxDnaConfig:
    truncation: float
    K: int
    threshold: float
    model_config: CompletionModelConfig
    ngrams_config: NGramsConfig
