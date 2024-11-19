import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, \
    RocCurveDisplay

from detectors.metrics import ClassificationMetrics, SplitConclusion, ClassificationRepresentations


def tpr_at_fpr_threshold(fpr, tpr, target_fpr=0.1):
    idx = np.where(fpr >= target_fpr)[0][0]

    if fpr[idx] == target_fpr:
        return tpr[idx]
    else:
        tpr_interp = tpr[idx - 1] + (tpr[idx] - tpr[idx - 1]) * (target_fpr - fpr[idx - 1]) / (fpr[idx] - fpr[idx - 1])
        return tpr_interp


def calculate_classification(y_true, y_scores) -> SplitConclusion:
    y_pred = y_scores > 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_pred)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot()

    mtrix = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    mtrix.plot()

    return SplitConclusion(metrics=ClassificationMetrics(
        tpr_at_1_percent_fpr=tpr_at_fpr_threshold(fpr, tpr, target_fpr=0.01),
        tpr_at_10_percent_fpr=tpr_at_fpr_threshold(fpr, tpr, target_fpr=0.1),
        auc=auc,
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        accuracy=accuracy_score(y_true, y_pred)
    ), representations=ClassificationRepresentations(
        roc_curve=display.figure_,
        clf_report=mtrix.figure_
    ))
