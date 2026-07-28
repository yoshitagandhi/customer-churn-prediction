"""
===============================================================================
Customer Churn Prediction Platform
Evaluation Service

File        : evaluation_service.py
Version     : 1.0

Purpose
-------
Provides the business-facing evaluation API for the Streamlit
application.

Responsibilities
----------------
• Model evaluation
• Model comparison
• Calibration analysis
• Learning curve generation
• Evaluation report generation

Notes
-----
• Never trains models.
• Never performs preprocessing.
• Never contains Streamlit code.
• Delegates all work to src.evaluation.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation import evaluate_model as backend_evaluate_model
from configs.logging_config import get_logger
from src.evaluation import (
    compare_models as backend_compare_models,
    generate_evaluation_report as backend_generate_report,
    generate_learning_curve,
    generate_calibration_curve,
    compute_metrics,
)
from src.utils.exceptions import (
    ConfigurationError,
    DataValidationError,
)
from src.evaluation.learning_curve import generate_learning_curve
from configs.config import settings

logger = get_logger(__name__)

@dataclass(slots=True)
class EvaluationResult:
    """Complete model evaluation."""

    metrics: dict[str, Any]
    calibration_curve_path: str | None
    learning_curve_path: str | None

@dataclass(slots=True)
class ComparisonResult:
    """Comparison between trained models."""

    best_model: str
    comparison_table: pd.DataFrame
    ranking: list[str]
    selection_reason: str

def _validate_model(model: Any) -> None:
    """Validate fitted model."""

    if model is None:
        raise ConfigurationError(
            "Prediction model has not been loaded."
        )

def _validate_features(features: pd.DataFrame) -> None:
    """Validate feature matrix."""

    if features.empty:
        raise DataValidationError(
            "Feature dataframe is empty."
        )

def _validate_target(target: pd.Series) -> None:
    """Validate target labels."""

    if target.empty:
        raise DataValidationError(
            "Target labels are empty."
        )


def _encode_target(target: pd.Series, positive_label: Any) -> pd.Series:
    """Encode target labels to binary (0/1) using positive_label."""

    if target.dtype == bool:
        return target.astype(int)

    # If labels already numeric, convert to 0/1 based on positive_label
    if pd.api.types.is_numeric_dtype(target):
        return (target == positive_label).astype(int)

    # For object/string labels
    return target.apply(lambda x: 1 if x == positive_label else 0).astype(int)

def evaluate_model(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a single fitted production model.
    Used by the Streamlit dashboard.
    """

    logger.info("Evaluating production model.")

    target_encoded = _encode_target(target, settings.positive_label)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics = compute_metrics(
        target_true=target_encoded.to_numpy(),
        target_pred=predictions,
        target_proba=probabilities,
    )

    result: dict[str, Any] = {"metrics": metrics}
    # Keep the row-level output alongside the aggregate metrics.  The
    # executive dashboard uses this to build the customer risk portfolio;
    # without it, it can only report that the portfolio is unavailable.
    result["predictions"] = pd.DataFrame(
        {
            "prediction": predictions,
            "churn_probability": probabilities,
        },
        index=features.index,
    )

    # Calibration and learning curves create files and the latter refits the
    # model repeatedly.  They are useful for a deliberate evaluation run, but
    # must not run whenever the landing page rerenders.
    if include_diagnostics:
        result["calibration"] = generate_calibration_curve(
            target_true=target_encoded.to_numpy(),
            target_proba=probabilities,
            model_name="Production Model",
        )
        result["learning_curve"] = generate_learning_curve(
            pipeline=model,
            features=features,
            target=target,
            model_name="Production Model",
        )

    return result

def compare_models(
    experiment_records: list[Any],
) -> ComparisonResult:
    """
    Compare trained models.
    """

    if not experiment_records:
        raise DataValidationError(
            "No experiment records were provided."
        )

    logger.info(
        "Comparing trained models."
    )

    comparison = backend_compare_models(
        experiment_records
    )

    logger.info(
        "Model comparison completed."
    )

    return ComparisonResult(
        best_model=comparison.best_model,
        comparison_table=comparison.comparison_table,
        ranking=comparison.ranking,
        selection_reason=comparison.selection_reason,
    )

def generate_evaluation_report(
    evaluation: EvaluationResult,
    output_directory: str,
) -> str:
    """
    Generate evaluation report.
    """
    
    logger.info(
        "Generating evaluation report."
    )

    report_path = backend_generate_report(
        evaluation=evaluation,
        output_directory=output_directory,
    )

    logger.info(
        "Evaluation report generated."
    )

    return report_path

def get_learning_curve(
    *,
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> str:
    """
    Generate learning curve.
    """
    _validate_model(model)
    _validate_features(features)
    _validate_target(target)

    return generate_learning_curve(
    pipeline=model,
    features=features,
    target=target,
    model_name="Production Model",
)

def get_calibration_curve(
    *,
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> str:
    """
    Generate calibration curve.
    """
    _validate_model(model)
    _validate_features(features)
    _validate_target(target)

    return generate_calibration_curve(
        model=model,
        features=features,
        target=target,
    )

__all__ = [
    "EvaluationResult",
    "ComparisonResult",
    "evaluate_model",
    "compare_models",
    "generate_evaluation_report",
    "get_learning_curve",
    "get_calibration_curve",
]
