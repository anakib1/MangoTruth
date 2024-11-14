
from dataclasses import dataclass
from typing import List

from matplotlib import pyplot as plt


@dataclass
class ClassificationMetrics:
    tpr_at_1_percent_fpr: float
    tpr_at_10_percent_fpr: float
    auc: float
    f1: float
    accuracy: float
    precision: float
    recall: float


@dataclass
class ClassificationRepresentations:
    roc_curve: plt.Figure
    clf_report: plt.Figure


@dataclass
class SplitConclusion:
    metrics: ClassificationMetrics
    representations: ClassificationRepresentations


@dataclass
class Conclusion:
    weights: bytes
    detector_handle: str
    datasets: List[str]
    train_conclusion: SplitConclusion
    validation_conclusion: SplitConclusion
