import numpy as np

MAX_DEPTH = 3

vector = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / np.where(y < 0, np.minimum(y, -0.001), np.maximum(y, 0.001)),
    lambda x, y: (x > y).astype(float),
    lambda x, y: (x < y).astype(float),
]


def top25mean(x: np.array):
    y = np.sort(x)
    return np.mean(y[-len(x) // 4:])


scalar = [
    lambda x: np.max(x),
    lambda x: np.min(x),
    lambda x: np.mean(x),
    top25mean,
    lambda x: np.linalg.norm(x, 1),
    lambda x: np.linalg.norm(x, 2),
    lambda x: np.mean((x - np.mean(x)) ** 2)
]


def find_all_features(depth, feat, probs, scalars, vectors):
    if depth > MAX_DEPTH:
        return []
    s = list()
    for f in scalars:
        s.append(f(feat))

    for ps in probs:
        for fv in vectors:
            s.extend(find_all_features(depth + 1, fv(feat, ps), probs, scalars, vectors))

    return s


def extract_features(feats):
    ret = []
    for feat in feats:
        ret.extend(find_all_features(1, feat, feats, scalar, vector))

    return np.array(ret)
