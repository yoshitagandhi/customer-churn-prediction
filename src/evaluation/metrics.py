"""Classification metric computation.

Pure functions only — no visualization, no orchestration. Every
metric is computed with a scikit-learn function wherever one exists.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from configs.logging_config import get_logger

logger = get_logger(__name__)


def compute_metrics(
    target_true: np.ndarray, target_pred: np.ndarray, target_proba: np.ndarray
) -> dict[str, Any]:
    """Compute the full set of classification metrics for one model.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_pred: Predicted labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.

    Returns:
        A dictionary with primary metrics (roc_auc, pr_auc, f1,
        precision, recall), secondary metrics (accuracy,
        balanced_accuracy, specificity, sensitivity, mcc), and
        confusion matrix statistics (true_positives, true_negatives,
        false_positives, false_negatives).
    """
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target_true, target_pred
    ).ravel()

    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0.0
    )
    sensitivity = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return {
        # Primary metrics
        "roc_auc": round(float(roc_auc_score(target_true, target_proba)), 4),
        "pr_auc": round(float(average_precision_score(target_true, target_proba)), 4),
        "f1": round(float(f1_score(target_true, target_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(target_true, target_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(target_true, target_pred, zero_division=0)), 4),
        # Secondary metrics (reference only; never used to pick the best model)
        "accuracy": round(float(accuracy_score(target_true, target_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(target_true, target_pred)), 4),
        "specificity": round(float(specificity), 4),
        "sensitivity": round(float(sensitivity), 4),
        "mcc": round(float(matthews_corrcoef(target_true, target_pred)), 4),
        # Confusion matrix statistics
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
    }
