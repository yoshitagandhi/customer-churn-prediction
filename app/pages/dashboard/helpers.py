"""
Customer Churn Prediction Platform
Dashboard Helpers

Shared helper functions used across the dashboard package.

This module contains reusable business logic only.
No Streamlit rendering should occur here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import (
    EXCELLENT_SCORE,
    GOOD_SCORE,
    WARNING_SCORE,
)

def determine_health_status(score: float) -> str:
    """
    Determine the overall model health category.

    Parameters
    ----------
    score:
        Model evaluation score.

    Returns
    -------
    str
        Health category.
    """

    if score >= EXCELLENT_SCORE:
        return "Excellent"

    if score >= GOOD_SCORE:
        return "Good"

    if score >= WARNING_SCORE:
        return "Warning"

    return "Critical"


def determine_deployment_status(score: float) -> str:
    """
    Determine deployment recommendation.

    Parameters
    ----------
    score:
        Model evaluation score.

    Returns
    -------
    str
        Deployment recommendation.
    """

    if score >= EXCELLENT_SCORE:
        return "Production Ready"

    if score >= GOOD_SCORE:
        return "Monitoring Recommended"

    return "Active Model"

def calculate_dataset_summary(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """
    Calculate dataset statistics.

    Parameters
    ----------
    features:
        Validation feature dataset.

    target:
        Validation labels.

    Returns
    -------
    dict
        Dataset summary.
    """

    total_records = len(features)

    total_features = features.shape[1]

    from configs.config import settings

    if not pd.api.types.is_numeric_dtype(target):
        churn_count = int((target == settings.positive_label).sum())
    else:
        churn_count = int(target.sum())

    retained_count = total_records - churn_count

    churn_rate = (
        churn_count / total_records
        if total_records
        else 0.0
    )

    missing_values = int(
        features.isna().sum().sum()
    )

    completeness = (
        1
        - (
            missing_values
            / (total_records * total_features)
        )
        if total_records and total_features
        else 0.0
    )

    return {
        "records": total_records,
        "features": total_features,
        "churn": churn_count,
        "retained": retained_count,
        "churn_rate": churn_rate,
        "missing_values": missing_values,
        "completeness": completeness,
    }

def safe_metadata(
    metadata: dict[str, Any] | None,
    key: str,
    default: Any = "N/A",
) -> Any:
    """
    Safely retrieve metadata values.

    Parameters
    ----------
    metadata:
        Metadata dictionary.

    key:
        Metadata key.

    default:
        Default value.

    Returns
    -------
    Any
    """

    if not metadata:
        return default

    return metadata.get(
        key,
        default,
    )

def get_metric(
    evaluation: Any,
    metric: str,
    default: float = 0.0,
) -> float:
    """
    Safely retrieve evaluation metrics.

    Parameters
    ----------
    evaluation:
        Evaluation result object.

    metric:
        Metric name.

    default:
        Default value.

    Returns
    -------
    float
    """

    return getattr(
        evaluation,
        metric,
        default,
    )

__all__ = [
    "calculate_dataset_summary",
    "determine_deployment_status",
    "determine_health_status",
    "get_metric",
    "safe_metadata",
]
