"""
===============================================================================
Customer Churn Prediction Platform
Prediction Service

File        : prediction_service.py
Version     : 1.0

Purpose
-------
Provides the business-facing prediction API for the Streamlit application.

Responsibilities
----------------
- Single customer prediction
- Batch prediction
- Prediction summaries
- Risk categorization

Notes
-----
- Never performs preprocessing.
- Never trains models.
- Never contains Streamlit code.
- Uses the serialized production pipeline.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.utils.exceptions import ConfigurationError, DataValidationError

logger = get_logger(__name__)

HIGH_RISK_THRESHOLD: Final[float] = 0.80
MEDIUM_RISK_THRESHOLD: Final[float] = 0.50

@dataclass(slots=True)
class PredictionResult:
    """Business-friendly prediction result."""

    predicted_class: str
    predicted_probability: float
    confidence: float
    risk_level: str
    recommended_action: str
    prediction_timestamp: str
    customer_frame: pd.DataFrame

def _validate_model(model: Any) -> None:
    """Validate the loaded prediction pipeline."""

    if model is None:
        raise ConfigurationError("Prediction model is not loaded.")

    if not hasattr(model, "predict"):
        raise ConfigurationError("Loaded model does not implement predict().")

    if not hasattr(model, "predict_proba"):
        raise ConfigurationError(
            "Loaded model does not implement predict_proba()."
        )
        
def _validate_customer(customer_data: dict[str, Any]) -> None:
    """Validate customer input."""

    if not customer_data:
        raise DataValidationError("Customer data cannot be empty.")


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate batch input."""

    if dataframe.empty:
        raise DataValidationError("Input dataframe is empty.")

def _determine_risk_level(probability: float) -> str:
    """Convert probability into business risk."""

    if probability >= HIGH_RISK_THRESHOLD:
        return "High"

    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"

    return "Low"

def _recommended_action(risk_level: str) -> str:
    """Business recommendation."""

    actions = {
        "High": (
            "Immediately engage the customer with a retention campaign."
        ),
        "Medium": (
            "Monitor customer behaviour and consider proactive incentives."
        ),
        "Low": (
            "Continue normal customer engagement."
        ),
    }

    return actions.get(
        risk_level,
        "Review customer profile."
    )

def _positive_class_probability(model: Any, probabilities: Any) -> float:
    """Return the probability of the churn/positive class.

    The trained IBM Telco pipeline uses numeric classes ``0`` and ``1``.
    For compatible string-labelled estimators, the configured positive label
    is used when available.
    """
    classes = list(getattr(model, "classes_", []))
    positive_candidates = (1, True, settings.positive_label)

    for label in positive_candidates:
        if label in classes:
            return float(probabilities[classes.index(label)])

    if len(probabilities) == 2:
        return float(probabilities[1])

    raise ConfigurationError("Unable to identify the model's positive class.")

def _prediction_probability(
    model: Any,
    prediction: Any,
    probabilities: Any,
) -> float:
    """Backward-compatible helper returning positive-class probability."""
    return _positive_class_probability(model, probabilities)

def predict_customer(
    customer_data: dict[str, Any],
    model: Any,
) -> PredictionResult:
    """
    Predict churn for a single customer.
    """

    logger.info("Starting customer prediction.")

    _validate_model(model)
    _validate_customer(customer_data)

    customer_frame = pd.DataFrame([customer_data])

    predicted_class = model.predict(customer_frame)[0]

    probabilities = model.predict_proba(customer_frame)[0]

    probability = _positive_class_probability(model, probabilities)

    risk_level = _determine_risk_level(probability)

    result = PredictionResult(
        predicted_class=str(predicted_class),
        predicted_probability=round(probability, 4),
        confidence=round(probability, 4),
        risk_level=risk_level,
        recommended_action=_recommended_action(risk_level),
        prediction_timestamp=datetime.now(
            UTC
        ).isoformat(),
        customer_frame=customer_frame,
    )

    logger.info("Customer prediction completed.")

    return result

def predict_batch(
    dataframe: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """
    Predict churn for multiple customers.
    """

    logger.info("Starting batch prediction.")

    _validate_model(model)
    _validate_dataframe(dataframe)

    result = dataframe.copy()

    predictions = model.predict(result)

    probabilities = model.predict_proba(result)

    predicted_probabilities = [
        _positive_class_probability(model, proba)
        for prediction, proba in zip(
            predictions,
            probabilities,
            strict=False,
        )
    ]

    timestamp = datetime.now(
        UTC
    ).isoformat()

    result["predicted_class"] = predictions
    result["predicted_probability"] = predicted_probabilities
    result["confidence"] = predicted_probabilities

    result["risk_level"] = result[
        "confidence"
    ].apply(_determine_risk_level)

    result["recommended_action"] = result[
        "risk_level"
    ].apply(_recommended_action)

    result["prediction_timestamp"] = timestamp

    logger.info("Batch prediction completed.")

    return result

def summarize_predictions(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """
    Generate dashboard summary statistics.
    """

    _validate_dataframe(dataframe)

    if "risk_level" not in dataframe.columns:
        raise DataValidationError(
            "Prediction results do not contain 'risk_level'."
        )

    return {
        "total_predictions": len(dataframe),
        "high_risk": int(
            (dataframe["risk_level"] == "High").sum()
        ),
        "medium_risk": int(
            (dataframe["risk_level"] == "Medium").sum()
        ),
        "low_risk": int(
            (dataframe["risk_level"] == "Low").sum()
        ),
    }

__all__ = [
    "PredictionResult",
    "predict_customer",
    "predict_batch",
    "summarize_predictions",
]