"""Threshold performance analysis.

Evaluates classification metrics and business cost across a
configurable range of thresholds, returning a single structured table
for optimization, reporting, and visualization. No plotting happens
here.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from configs.config import settings
from configs.logging_config import get_logger
from src.threshold.cost_analysis import calculate_business_cost

logger = get_logger(__name__)


def _compute_metrics_at_threshold(
    target_true: np.ndarray, target_proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Compute classification metrics and business cost at a single threshold.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.
        threshold: Classification threshold to apply.

    Returns:
        A dictionary with precision, recall, F1, specificity, false
        positive/negative rates, confusion matrix counts, and business
        cost figures.
    """
    target_pred = (target_proba >= threshold).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target_true, target_pred, labels=[0, 1]
    ).ravel()

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0.0
    )
    false_positive_rate = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )
    false_negative_rate = (
        false_negatives / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0.0
    )

    cost_breakdown = calculate_business_cost(
        true_positives, false_positives, true_negatives, false_negatives
    )

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        **cost_breakdown,
    }


def evaluate_thresholds(
    target_true: np.ndarray,
    target_proba: np.ndarray,
    threshold_range: tuple[float, float] = settings.threshold_range,
    threshold_step: float = settings.threshold_step,
) -> pd.DataFrame:
    """Evaluate metrics and business cost across a range of thresholds.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.
        threshold_range: (min, max) thresholds to evaluate, inclusive.
            Defaults to ``settings.threshold_range``.
        threshold_step: Step size between evaluated thresholds.
            Defaults to ``settings.threshold_step``.

    Returns:
        A DataFrame with one row per threshold, including precision,
        recall, F1, specificity, false positive/negative rates, and
        business cost.
    """
    lower_bound, upper_bound = threshold_range
    thresholds = np.round(
        np.arange(lower_bound, upper_bound + threshold_step / 2, threshold_step), 4
    )
    rows = [
        _compute_metrics_at_threshold(target_true, target_proba, threshold)
        for threshold in thresholds
    ]
    logger.info("Threshold evaluation completed: %d threshold(s) evaluated.", len(rows))
    return pd.DataFrame(rows)
