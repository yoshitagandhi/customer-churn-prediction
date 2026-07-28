"""Business-friendly interpretation of SHAP explanations.

Converts a technical prediction explanation (feature names, SHAP
values) into plain language and recommendations a non-technical
stakeholder can act on. Every recommendation is derived only from the
features that actually appeared as contributors for that specific
prediction — nothing here is a generic, unsupported suggestion.
"""

import re
from typing import Any, Final

from configs.logging_config import get_logger

logger = get_logger(__name__)

# Risk-level thresholds applied to the predicted churn probability.
_HIGH_RISK_THRESHOLD: Final[float] = 0.6
_MEDIUM_RISK_THRESHOLD: Final[float] = 0.3

# Maps a feature's base name (before any "_Category" one-hot suffix)
# to a business action. Only features that actually appear among a
# prediction's top positive contributors trigger a recommendation —
# this dict never introduces a suggestion unsupported by the SHAP
# explanation itself.
_RECOMMENDATION_RULES: Final[dict[str, str]] = {
    "Contract": "Offer a long-term contract discount to encourage commitment.",
    "ContractRisk": "Offer a long-term contract discount to encourage commitment.",
    "tenure": "Prioritize proactive retention outreach for newer customers.",
    "TenureGroup": "Prioritize proactive retention outreach for newer customers.",
    "MonthlyCharges": "Review pricing or offer a loyalty discount to reduce cost sensitivity.",
    "AverageMonthlySpend": "Review pricing or offer a loyalty discount to reduce cost sensitivity.",
    "TechSupport": "Recommend bundling technical support services.",
    "OnlineSecurity": "Recommend bundling online security services.",
    "OnlineBackup": "Recommend bundling online backup services.",
    "DeviceProtection": "Recommend bundling device protection services.",
    "InternetService": "Review internet service plan options with the customer.",
    "PaymentMethod": "Encourage a switch to automatic/electronic payment methods.",
    "PaperlessBilling": "Encourage enrollment in paperless billing.",
    "StreamingTV": "Recommend bundling streaming service add-ons.",
    "StreamingMovies": "Recommend bundling streaming service add-ons.",
}

_DEFAULT_RECOMMENDATION: Final[str] = (
    "Schedule a follow-up engagement to better understand this customer's needs."
)


def _humanize_feature_name(feature_name: str) -> str:
    """Convert an encoded feature name into a readable phrase.

    Examples: "Contract_Month-to-month" -> "Contract: Month-to-month";
    "MonthlyCharges" -> "Monthly Charges".

    Args:
        feature_name: A processed (possibly one-hot-encoded) column
            name.

    Returns:
        A human-readable phrase, with no SHAP or encoding jargon.
    """
    base_name, _, category = feature_name.partition("_")
    readable_base = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", base_name)
    readable_base = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", readable_base)
    return f"{readable_base}: {category}" if category else readable_base


def _generate_recommendations(top_positive_contributors: list[dict[str, Any]]) -> list[str]:
    """Derive recommendations strictly from a prediction's own top contributors.

    Args:
        top_positive_contributors: Output of
            ``prediction_explanation["top_positive_contributors"]``.

    Returns:
        A list of recommendation strings, one per distinct rule
        matched, in the same order as the contributors.
    """
    recommendations: list[str] = []
    seen_rules: set[str] = set()

    for contributor in top_positive_contributors:
        base_feature = contributor["feature"].split("_")[0]
        rule = _RECOMMENDATION_RULES.get(base_feature)
        if rule and rule not in seen_rules:
            recommendations.append(rule)
            seen_rules.add(rule)

    if not recommendations:
        recommendations.append(_DEFAULT_RECOMMENDATION)
    return recommendations


def generate_business_insights(prediction_explanation: dict[str, Any]) -> dict[str, Any]:
    """Translate a technical prediction explanation into business language.

    Args:
        prediction_explanation: Output of
            :func:`src.explainability.prediction_explainer.explain_prediction`.

    Returns:
        A dictionary with the risk level, plain-language main reasons
        and protective factors, grounded recommendations, and a
        jargon-free narrative summary.
    """
    probability = prediction_explanation["predicted_probability"]

    if probability >= _HIGH_RISK_THRESHOLD:
        risk_level = "High"
    elif probability >= _MEDIUM_RISK_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    main_reasons = [
        _humanize_feature_name(item["feature"])
        for item in prediction_explanation["top_positive_contributors"]
    ]
    protective_factors = [
        _humanize_feature_name(item["feature"])
        for item in prediction_explanation["top_negative_contributors"]
    ]
    recommendations = _generate_recommendations(prediction_explanation["top_positive_contributors"])

    narrative = (
        f"This customer has a {probability:.0%} predicted probability of churning "
        f"({risk_level} risk)."
    )
    if main_reasons:
        narrative += f" The main factors increasing risk are: {', '.join(main_reasons)}."
    if protective_factors:
        narrative += f" Factors reducing risk include: {', '.join(protective_factors)}."

    logger.debug("Business insights generated: risk_level=%s.", risk_level)

    return {
        "risk_level": risk_level,
        "predicted_probability": probability,
        "main_reasons": main_reasons,
        "protective_factors": protective_factors,
        "recommendations": recommendations,
        "narrative": narrative,
    }
