import numpy as np


def safe_sigmoid(x):
    positive = x >= 0
    negative = ~positive
    result = np.zeros_like(x, dtype=np.float64)

    result[positive] = 1 / (1 + np.exp(-x[positive]))
    result[negative] = np.exp(x[negative]) / (1 + np.exp(x[negative]))

    return result
