"""Feature importance summaries.

Turns raw SHAP output (from :mod:`src.explainability.shap_analysis`)
into ranked feature lists reusable by reports and the future
Streamlit dashboard. This module performs no SHAP computation itself.
"""

from typing import Any

import numpy as np

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


def rank_features(
    global_explanation: dict[str, Any], top_n: int = settings.shap_top_n_features
) -> list[dict[str, Any]]:
    """Rank features by global mean |SHAP| value.

    Args:
        global_explanation: Output of
            :func:`src.explainability.shap_analysis.generate_global_explanations`.
        top_n: How many top features to return. Defaults to
            ``settings.shap_top_n_features``.

    Returns:
        A list of dictionaries, each with "rank", "feature", and
        "mean_abs_shap", ordered from most to least influential.
    """
    ranked = [
        {
            "rank": index + 1,
            "feature": feature,
            "mean_abs_shap": global_explanation["mean_abs_shap"][feature],
        }
        for index, feature in enumerate(global_explanation["feature_ranking"][:top_n])
    ]
    logger.debug("Ranked top %d feature(s) by global importance.", len(ranked))
    return ranked


def get_top_positive_contributors(
    shap_values_array: np.ndarray,
    feature_names: list[str],
    top_n: int = settings.shap_top_n_features,
) -> list[dict[str, Any]]:
    """Identify features that, on average, push predictions toward churn.

    Args:
        shap_values_array: A 2D array of SHAP values (rows x features).
        feature_names: Feature names, in the same column order as
            ``shap_values_array``.
        top_n: How many top contributors to return. Defaults to
            ``settings.shap_top_n_features``.

    Returns:
        A list of dictionaries with "feature" and "mean_shap_value",
        limited to features with a positive average contribution,
        ordered from strongest to weakest.
    """
    mean_signed = shap_values_array.mean(axis=0)
    positive = sorted(
        (
            {"feature": name, "mean_shap_value": round(float(value), 6)}
            for name, value in zip(feature_names, mean_signed, strict=True)
            if value > 0
        ),
        key=lambda item: -item["mean_shap_value"],
    )
    return positive[:top_n]


def get_top_negative_contributors(
    shap_values_array: np.ndarray,
    feature_names: list[str],
    top_n: int = settings.shap_top_n_features,
) -> list[dict[str, Any]]:
    """Identify features that, on average, push predictions toward retention.

    Args:
        shap_values_array: A 2D array of SHAP values (rows x features).
        feature_names: Feature names, in the same column order as
            ``shap_values_array``.
        top_n: How many top contributors to return. Defaults to
            ``settings.shap_top_n_features``.

    Returns:
        A list of dictionaries with "feature" and "mean_shap_value",
        limited to features with a negative average contribution,
        ordered from strongest to weakest (most negative first).
    """
    mean_signed = shap_values_array.mean(axis=0)
    negative = sorted(
        (
            {"feature": name, "mean_shap_value": round(float(value), 6)}
            for name, value in zip(feature_names, mean_signed, strict=True)
            if value < 0
        ),
        key=lambda item: item["mean_shap_value"],
    )
    return negative[:top_n]
