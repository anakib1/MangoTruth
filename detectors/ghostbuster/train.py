import pathlib
from pickle import dump

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegressionCV
from tqdm.auto import tqdm

from detectors.ghostbuster.features import extract_features
from detectors.ghostbuster.ngrams import UnigramModel, TrigramModel

if __name__ == '__main__':
    xlsum = load_dataset('anakib1/mango-truth', 'xlsum')
    reference = xlsum['train'].filter(lambda x: x['label'] == 0).take(5000)
    generated = xlsum['train'].filter(lambda x: x['label'] == 3)

    X = reference['output'] + generated['output']
    y = [0] * len(reference) + [1] * len(generated)

    unigram = UnigramModel()
    unigram.train(''.join(X))

    trigram = TrigramModel()
    trigram.train(''.join(X))

    feats = np.array([extract_features([unigram.predict_proba(x), trigram.predict_proba(x)]) for x in tqdm(X)])

    model = LogisticRegressionCV()
    model.fit(feats, y)

    pathlib.Path('bin').mkdir(exist_ok=True)
    with open("bin/clf.pkl", "wb") as f:
        dump(model, f, protocol=5)

    with open("bin/unigram.pkl", "wb") as f:
        dump(unigram, f, protocol=5)

    with open("bin/trigram.pkl", "wb") as f:
        dump(trigram, f, protocol=5)
