from dataclasses import dataclass
from nltk.stem import StemmerI
from nltk.stem.porter import PorterStemmer
from detectors.dna.functions import DnaSerializableFunction, SublogarithmFunction
from detectors.interfaces import SerializableConfig
import spacy
from detectors.models.config import CompletionModelConfig


@dataclass
class NGramsConfig(SerializableConfig):
    N_min: int
    N_max: int
    stemmer: StemmerI
    func: DnaSerializableFunction
    stopwords: list[str]


class NGramsConfigGenerator(NGramsConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__(
                N_min=4,
                N_max=25,
                stemmer=PorterStemmer(),
                func=SublogarithmFunction(),
                stopwords=spacy.load("en_core_web_sm").Defaults.stop_words
            )
        else:
            raise NotImplementedError("There are no such parameter set name")


@dataclass
class BlackBoxDnaConfig(SerializableConfig):
    truncation: float
    K: int
    threshold: float
    model_config: CompletionModelConfig
    ngrams_config: NGramsConfig
