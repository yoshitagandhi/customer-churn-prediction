"""Recommendation service.

Bridges the Streamlit UI to Milestone 9's threshold optimization and
business decision engine. No threshold-optimization logic is
reimplemented here — the optimized threshold is loaded from disk, not
recomputed.
"""

from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.threshold.decision_engine import DEFAULT_RISK_BANDS, classify_risk

logger = get_logger(__name__)


def generate_recommendation(
    probability: float,
    threshold_config: dict[str, Any],
    positive_label: str = settings.positive_label,
) -> dict[str, Any]:
    """Classify a single prediction into a risk level and recommended action.

    Args:
        probability: Predicted probability of churn.
        threshold_config: The optimized threshold configuration (see
            :func:`app.utils.cache.get_cached_threshold_config`).
        positive_label: Label representing churn. Defaults to
            ``settings.positive_label``.

    Returns:
        A dictionary with the predicted class, risk level, and
        recommended action for this customer.
    """
    optimal_threshold = threshold_config["optimal_threshold"]
    predicted_class = positive_label if probability >= optimal_threshold else "No"
    risk_band = classify_risk(probability)

    return {
        "predicted_class": predicted_class,
        "predicted_probability": probability,
        "threshold_used": optimal_threshold,
        "risk_level": risk_band.name,
        "recommended_action": risk_band.action,
    }


def generate_batch_recommendations(
    dataframe: pd.DataFrame,
    threshold_config: dict[str, Any],
    positive_label: str = settings.positive_label,
) -> pd.DataFrame:
    """Generate risk levels and recommended actions for a batch of predictions.

    Args:
        dataframe: A DataFrame with a "predicted_probability" column
            (see :func:`app.services.prediction_service.predict_batch`).
        threshold_config: The optimized threshold configuration.
        positive_label: Label representing churn. Defaults to
            ``settings.positive_label``.

    Returns:
        A copy of ``dataframe`` with "predicted_class", "risk_level",
        and "recommended_action" columns appended.
    """
    optimal_threshold = threshold_config["optimal_threshold"]
    result = dataframe.copy()
    # Determine predicted class based on the optimized threshold
    result["predicted_class"] = result["predicted_probability"].apply(
        lambda probability: positive_label if probability >= optimal_threshold else "No"
    )
    bands = result["predicted_probability"].apply(classify_risk)
    result["risk_level"] = [band.name for band in bands]
    result["recommended_action"] = [band.action for band in bands]
    return result


__all__ = ["generate_recommendation", "generate_batch_recommendations", "DEFAULT_RISK_BANDS"]
