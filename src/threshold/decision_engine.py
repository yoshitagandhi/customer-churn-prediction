"""Business decision engine.

Converts prediction probabilities and the optimized threshold into
structured, actionable business recommendations — never plain text.
Risk bands and their associated actions are configurable rather than
fixed strings baked into the logic.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from configs.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskBand:
    """A single configurable business risk segment.

    Attributes:
        name: Risk level label (e.g., "Low", "Medium", "High").
        lower_bound: Inclusive lower probability bound.
        upper_bound: Exclusive upper probability bound.
        action: Recommended business action for this band.
    """

    name: str
    lower_bound: float
    upper_bound: float
    action: str


# Default risk segmentation. Fully overridable by passing a custom
# `risk_bands` tuple to `classify_risk` or `generate_recommendations`.
DEFAULT_RISK_BANDS: tuple[RiskBand, ...] = (
    RiskBand("Low", 0.0, 0.2, "No Action Required"),
    RiskBand("Medium", 0.2, 0.5, "Send Promotional Offer"),
    RiskBand("High", 0.5, 0.8, "Assign Retention Team"),
    RiskBand("Very High", 0.8, 1.01, "Escalate to Customer Success"),
)


def classify_risk(
    probability: float, risk_bands: tuple[RiskBand, ...] = DEFAULT_RISK_BANDS
) -> RiskBand:
    """Classify a single predicted probability into a business risk band.

    Args:
        probability: Predicted probability of churn.
        risk_bands: Risk band definitions to classify against.
            Defaults to :data:`DEFAULT_RISK_BANDS`.

    Returns:
        The matching RiskBand (the last band is used as a fallback if
        no band's bounds match, e.g. for probability == 1.0 with a
        strict upper bound).
    """
    for band in risk_bands:
        if band.lower_bound <= probability < band.upper_bound:
            return band
    return risk_bands[-1]


def generate_recommendations(
    probabilities: pd.Series,
    optimal_threshold: float,
    customer_ids: pd.Index | None = None,
    positive_label: str = "Yes",
    risk_bands: tuple[RiskBand, ...] = DEFAULT_RISK_BANDS,
) -> pd.DataFrame:
    """Generate structured business recommendations for a set of customers.

    Args:
        probabilities: Predicted churn probabilities.
        optimal_threshold: The classification threshold (from
            :func:`src.threshold.optimizer.optimize_threshold`) used
            to derive the predicted class.
        customer_ids: Identifiers for each customer, aligned with
            ``probabilities``. Defaults to ``probabilities.index``.
        positive_label: The label representing churn. Defaults to "Yes".
        risk_bands: Risk band definitions. Defaults to
            :data:`DEFAULT_RISK_BANDS`.

    Returns:
        A DataFrame with one row per customer: customer_id,
        predicted_probability, predicted_class, risk_level, and
        recommended_action.
    """
    ids = customer_ids if customer_ids is not None else probabilities.index

    records: list[dict[str, Any]] = []
    for customer_id, probability in zip(ids, probabilities, strict=True):
        band = classify_risk(float(probability), risk_bands)
        predicted_class = positive_label if probability >= optimal_threshold else "No"
        records.append(
            {
                "customer_id": customer_id,
                "predicted_probability": round(float(probability), 4),
                "predicted_class": predicted_class,
                "risk_level": band.name,
                "recommended_action": band.action,
            }
        )

    logger.debug("Generated %d business recommendation(s).", len(records))
    return pd.DataFrame(records)
