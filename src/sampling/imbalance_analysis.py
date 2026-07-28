"""Class imbalance analysis.

This module only *analyzes* a target variable's class distribution —
it never resamples, modifies, or otherwise touches the underlying
dataset. Resampling lives in :mod:`src.sampling.samplers` and
:mod:`src.sampling.pipeline`.
"""

from typing import Any, Final

import pandas as pd

from src.utils.exceptions import DataValidationError
from configs.logging_config import get_logger

logger = get_logger(__name__)

# Below this majority:minority ratio, classes are considered close
# enough to balanced that sampling is optional rather than advisable.
IMBALANCE_RATIO_THRESHOLD: Final[float] = 1.5


def analyze_class_distribution(target: pd.Series) -> dict[str, Any]:
    """Analyze the class distribution of a target (label) variable.

    Args:
        target: The target Series to analyze (e.g., a training
            fold's labels). Never modified.

    Returns:
        A dictionary with total sample count, per-class counts and
        percentages, majority/minority class identification, the
        imbalance ratio, whether sampling is recommended, and a
        human-readable summary.
    """
    logger.info("Sampling analysis started.")
    
    if target.empty:
        raise DataValidationError(
            "Target series cannot be empty."
    )

    value_counts = target.value_counts(dropna=False)
    logger.debug(
        "Detected %d unique class(es).",
         len(value_counts),
)
    total_samples = len(target)

    class_counts = {str(label): int(count) for label, count in value_counts.items()}
    class_percentages = {
        str(label): round(float(count) / total_samples * 100, 2)
        for label, count in value_counts.items()
    }

    majority_class = str(value_counts.idxmax())
    minority_class = str(value_counts.idxmin())
    if len(value_counts) == 1:
        logger.warning(
            "Only one target class detected."
    )
    majority_count = int(value_counts.max())
    minority_count = int(value_counts.min())
    imbalance_ratio = (
        round(majority_count / minority_count, 2) if minority_count > 0 else float("inf")
    )
    sampling_recommended = imbalance_ratio >= IMBALANCE_RATIO_THRESHOLD

    recommendation_text = (
        "Sampling is recommended to address this imbalance."
        if sampling_recommended
        else "Classes are reasonably balanced; sampling is optional."
    )
    summary = (
        f"Majority class '{majority_class}' ({majority_count} samples, "
        f"{class_percentages[majority_class]}%) vs. minority class '{minority_class}' "
        f"({minority_count} samples, {class_percentages[minority_class]}%). "
        f"Imbalance ratio: {imbalance_ratio}:1. {recommendation_text}"
    )

    logger.info("Class distribution calculated: imbalance ratio %s:1.", imbalance_ratio)

    return {
        "total_samples": total_samples,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "majority_count": majority_count,
        "minority_count": minority_count,
        "imbalance_ratio": imbalance_ratio,
        "sampling_recommended": sampling_recommended,
        "summary": summary,
    }
