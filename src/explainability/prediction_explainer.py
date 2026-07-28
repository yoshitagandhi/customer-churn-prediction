"""Individual prediction explanation.

Explains a single customer's prediction end to end: runs it through
the fitted pipeline, computes its SHAP contributions, and returns a
structured result reusable by reports and the future Streamlit
application.
"""

from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.explainability.explainer import get_processed_features
from src.explainability.shap_analysis import extract_positive_class_values

logger = get_logger(__name__)


def explain_prediction(
    pipeline: Any,
    explainer: Any,
    customer_features: pd.DataFrame,
    positive_label: str = settings.positive_label,
    top_n: int = settings.shap_top_n_features,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Explain a single customer's prediction.

    Args:
        pipeline: The fitted preprocessing + [sampling] + model
            pipeline.
        explainer: A SHAP Explainer instance from
            :func:`src.explainability.explainer.load_explainer`.
        customer_features: A single-row DataFrame of raw (pre-
            preprocessing) feature values for one customer. Any
            customer record can be passed — this is not limited to a
            fixed example.
        positive_label: The label representing churn. Defaults to
            ``settings.positive_label``.
        top_n: How many top positive/negative contributors to
            include. Defaults to ``settings.shap_top_n_features``.
        decision_threshold: Probability threshold used to derive the
            predicted class label. Threshold *optimization* is
            Milestone 9's responsibility; this default is a neutral
            0.5 midpoint, not a tuned business threshold.

    Returns:
        A dictionary with the predicted probability and class, the
        top positive/negative contributing features, and a plain-text
        explanation summary.
    """
    processed_row = get_processed_features(pipeline, customer_features)
    model = pipeline.named_steps["model"]

    predicted_probability = float(model.predict_proba(processed_row)[0, 1])
    predicted_class = positive_label if predicted_probability >= decision_threshold else "No"

    shap_values = explainer(processed_row)
    row_contributions = extract_positive_class_values(shap_values)[0]
    feature_names = processed_row.columns.tolist()

    contributions = sorted(
        zip(feature_names, row_contributions, strict=True), key=lambda pair: pair[1], reverse=True
    )
    top_positive = [
        {"feature": name, "shap_value": round(float(value), 6)}
        for name, value in contributions
        if value > 0
    ][:top_n]
    top_negative = sorted(
        (
            {"feature": name, "shap_value": round(float(value), 6)}
            for name, value in contributions
            if value < 0
        ),
        key=lambda item: item["shap_value"],
    )[:top_n]

    summary = _build_explanation_summary(
        predicted_probability, predicted_class, top_positive, top_negative
    )
    logger.debug(
        "Explained prediction: probability=%.4f, class=%s.", predicted_probability, predicted_class
    )

    return {
        "predicted_probability": round(predicted_probability, 4),
        "predicted_class": predicted_class,
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative,
        "explanation_summary": summary,
    }


def _build_explanation_summary(
    predicted_probability: float,
    predicted_class: str,
    top_positive: list[dict[str, Any]],
    top_negative: list[dict[str, Any]],
) -> str:
    """Build a concise, still-technical explanation summary sentence.

    A fully business-friendly (jargon-free) narrative is generated
    separately by :mod:`src.explainability.business_insights`; this
    summary is the technical-audience counterpart.

    Args:
        predicted_probability: Predicted probability of churn.
        predicted_class: Predicted class label.
        top_positive: Top positive (churn-increasing) contributors.
        top_negative: Top negative (churn-decreasing) contributors.

    Returns:
        A one-paragraph summary string.
    """
    positive_names = ", ".join(item["feature"] for item in top_positive[:3]) or "none identified"
    negative_names = ", ".join(item["feature"] for item in top_negative[:3]) or "none identified"
    return (
        f"Predicted class: {predicted_class} (probability={predicted_probability:.4f}). "
        f"Top features increasing churn risk: {positive_names}. "
        f"Top features decreasing churn risk: {negative_names}."
    )
