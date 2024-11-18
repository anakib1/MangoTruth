import numpy as np
from typing import List

MAX_DEPTH = 3

vector = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / np.where(y < 0, np.minimum(y, -0.001), np.maximum(y, 0.001)),
    lambda x, y: (x > y).astype(float),
    lambda x, y: (x < y).astype(float),
]

scalar = [
    lambda x: np.max(x),
    lambda x: np.min(x),
    lambda x: np.mean(x),
    lambda x: np.mean(np.sort(x)[-len(x) // 4:]),
    lambda x: np.linalg.norm(x, 1),
    lambda x: np.linalg.norm(x, 2),
    lambda x: np.mean((x - np.mean(x)) ** 2)
]


def _find_all_features(depth, feat, probs, scalars, vectors):
    if depth > MAX_DEPTH:
        return []
    s = list()
    for f in scalars:
        s.append(f(feat))

    for ps in probs:
        for fv in vectors:
            s.extend(_find_all_features(depth + 1, fv(feat, ps), probs, scalars, vectors))

    return s


def extract_features(feats: List[np.array]) -> np.array:
    ln = min([len(feat) for feat in feats])
    for i in range(len(feats)):
        feat = feats[i]
        if len(feat) != ln:
            print('WARNING: Not all length are equal. Got {} and {}'.format(ln, len(feat)))
        feats[i] = feat[:ln]

    ret = []
    for feat in feats:
        ret.extend(_find_all_features(1, feat, feats, scalar, vector))

    return np.array(ret)
