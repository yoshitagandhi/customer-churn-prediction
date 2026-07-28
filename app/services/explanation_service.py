"""
===============================================================================
Customer Churn Prediction Platform
Explainability Service

File        : explainability_service.py
Version     : 1.0

Purpose
-------
Provides the business-facing explainability API for the Streamlit
application.

Responsibilities
----------------
• Explain individual customer predictions
• Generate business insights
• Generate SHAP waterfall visualizations

Notes
-----
• Never performs prediction.
• Never computes SHAP values directly.
• Never contains Streamlit code.
• Delegates all explainability work to src.explainability.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

from configs.config import settings
from configs.logging_config import get_logger
from src.explainability import (
    explain_prediction as backend_explain_prediction,
    generate_business_insights,
    plot_shap_waterfall,
)
from src.explainability.explainer import (
    compute_shap_values,
    get_processed_features,
)
from src.utils.exceptions import (
    ConfigurationError,
    DataValidationError,
)

logger = get_logger(__name__)


# =============================================================================
# Result Models
# =============================================================================


@dataclass(slots=True)
class ExplanationResult:
    """
    Business-friendly prediction explanation.
    """
    prediction: Any
    business_insights: Any
    waterfall_path: str | None = None

def _validate_model(model: Any) -> None:
    """
    Validate prediction pipeline.
    """

    if model is None:
        raise ConfigurationError(
            "Prediction model has not been loaded."
        )

def _validate_explainer(explainer: Any) -> None:
    """
    Validate SHAP explainer.
    """
    if explainer is None:
        raise ConfigurationError(
            "SHAP explainer has not been initialized."
        )

def _validate_customer_data(
    customer_data: dict[str, Any],
) -> None:
    """
    Validate customer input.
    """
    if not customer_data:
        raise DataValidationError(
            "Customer data cannot be empty."
        )

def explain_customer_prediction(
    model: Any,
    explainer: Any,
    customer_data: dict[str, Any],
    *,
    positive_label: str = settings.positive_label,
) -> ExplanationResult:
    """
    Generate a complete explanation for one customer prediction.

    Parameters
    ----------
    model
        Loaded production prediction pipeline.

    explainer
        Cached SHAP explainer.

    customer_data
        Raw customer features.

    positive_label
        Positive churn label.

    Returns
    -------
    ExplanationResult
    """

    logger.info(
        "Generating customer explanation."
    )

    _validate_model(model)
    _validate_explainer(explainer)
    _validate_customer_data(customer_data)

    customer_frame = pd.DataFrame(
        [customer_data]
    )

    prediction = backend_explain_prediction(
        pipeline=model,
        explainer=explainer,
        customer_features=customer_frame,
        positive_label=positive_label,
    )

    business_insights = (
        generate_business_insights(
            prediction
        )
    )

    logger.info(
        "Customer explanation completed."
    )

    return ExplanationResult(
        prediction=prediction,
        business_insights=business_insights,
    )


def explain_customer_prediction_with_scenarios(
    model: Any,
    customer_data: dict[str, Any],
    reference_features: pd.DataFrame,
    *,
    positive_label: str = settings.positive_label,
    top_n: int = settings.shap_top_n_features,
) -> ExplanationResult:
    """Explain a prediction by varying one customer feature at a time.

    This lightweight local explanation is used when SHAP is not installed.
    Each contribution is the change in churn probability when a submitted
    feature replaces its typical value in the reference customer population.
    """

    _validate_model(model)
    _validate_customer_data(customer_data)
    if reference_features.empty:
        raise DataValidationError("Reference features are unavailable for explanation.")

    customer_frame = pd.DataFrame([customer_data])
    baseline_values: dict[str, Any] = {}
    for column in customer_frame.columns:
        values = reference_features[column].dropna()
        if values.empty:
            baseline_values[column] = customer_data[column]
        elif pd.api.types.is_numeric_dtype(values):
            baseline_values[column] = values.median()
        else:
            baseline_values[column] = values.mode().iat[0]
    baseline = pd.DataFrame([baseline_values], columns=customer_frame.columns)

    base_probability = float(np.asarray(model.predict_proba(baseline))[0, 1])
    predicted_probability = float(np.asarray(model.predict_proba(customer_frame))[0, 1])
    contributors: list[dict[str, Any]] = []
    for feature, value in customer_data.items():
        scenario = baseline.copy()
        scenario.at[scenario.index[0], feature] = value
        impact = float(np.asarray(model.predict_proba(scenario))[0, 1]) - base_probability
        contributors.append(
            {"feature": feature, "feature_value": value, "shap_value": round(impact, 6)}
        )

    ranked = sorted(contributors, key=lambda item: item["shap_value"], reverse=True)
    top_positive = [item for item in ranked if item["shap_value"] > 0][:top_n]
    top_negative = sorted(
        (item for item in ranked if item["shap_value"] < 0),
        key=lambda item: item["shap_value"],
    )[:top_n]
    prediction = {
        "predicted_probability": round(predicted_probability, 4),
        "predicted_class": positive_label if predicted_probability >= 0.5 else "No",
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative,
        "explanation_method": "Scenario analysis",
        "explanation_summary": (
            "Feature impacts compare this customer with a typical customer profile; "
            "positive values increase predicted churn risk."
        ),
    }
    return ExplanationResult(
        prediction=prediction,
        business_insights=generate_business_insights(prediction),
    )

def generate_waterfall_figure(
    model: Any,
    explainer: Any,
    customer_data: dict[str, Any],
) -> str:
    """
    Generate a SHAP waterfall visualization for one customer.
    """

    logger.info(
        "Generating SHAP waterfall."
    )

    _validate_model(model)
    _validate_explainer(explainer)
    _validate_customer_data(customer_data)

    customer_frame = pd.DataFrame(
        [customer_data]
    )

    processed_features = (
        get_processed_features(
            model,
            customer_frame,
        )
    )

    shap_values = compute_shap_values(
        explainer,
        processed_features,
    )

    figure_path = plot_shap_waterfall(
        shap_values=shap_values,
        row_index=0,
    )

    logger.info(
        "SHAP waterfall generated."
    )

    return figure_path

def explain_with_visualization(
    model: Any,
    explainer: Any,
    customer_data: dict[str, Any],
    *,
    positive_label: str = settings.positive_label,
) -> ExplanationResult:
    """
    Generate both explanation and visualization.

    This is a convenience function used by the
    Prediction page.
    """
    explanation = explain_customer_prediction(
        model=model,
        explainer=explainer,
        customer_data=customer_data,
        positive_label=positive_label,
    )

    waterfall = generate_waterfall_figure(
        model=model,
        explainer=explainer,
        customer_data=customer_data,
    )

    explanation.waterfall_path = waterfall

    return explanation

__all__ = [
    "ExplanationResult",
    "explain_customer_prediction",
    "explain_customer_prediction_with_scenarios",
    "generate_waterfall_figure",
    "explain_with_visualization",
]
