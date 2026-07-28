"""Threshold optimization, business cost analysis, and decision engine.

Transforms the trained model's prediction probabilities (Milestone
6/7) into a business decision system: finds the classification
threshold that best satisfies a configurable business objective,
estimates business cost/savings, and converts probabilities into
structured, actionable recommendations. No model retraining or SHAP
computation happens here.

Typical usage::

    from src.threshold import run_threshold_optimization_pipeline

    results = run_threshold_optimization_pipeline(target_true, target_proba)
"""

from typing import Any

import numpy as np
import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.threshold.cost_analysis import calculate_business_cost
from src.threshold.decision_engine import (
    DEFAULT_RISK_BANDS,
    RiskBand,
    classify_risk,
    generate_recommendations,
)
from src.threshold.optimizer import (
    SUPPORTED_OBJECTIVES,
    load_threshold_config,
    optimize_threshold,
    save_threshold_config,
)
from src.threshold.report import generate_threshold_report
from src.threshold.threshold_analysis import evaluate_thresholds
from src.threshold.visualization import (
    plot_business_cost_curve,
    plot_optimal_confusion_matrix,
    plot_precision_recall_tradeoff,
    plot_threshold_metrics,
)

logger = get_logger(__name__)

__all__ = [
    "calculate_business_cost",
    "evaluate_thresholds",
    "optimize_threshold",
    "save_threshold_config",
    "load_threshold_config",
    "SUPPORTED_OBJECTIVES",
    "RiskBand",
    "DEFAULT_RISK_BANDS",
    "classify_risk",
    "generate_recommendations",
    "plot_threshold_metrics",
    "plot_business_cost_curve",
    "plot_precision_recall_tradeoff",
    "plot_optimal_confusion_matrix",
    "generate_threshold_report",
    "run_threshold_optimization_pipeline",
]


def run_threshold_optimization_pipeline(
    target_true: np.ndarray,
    target_proba: np.ndarray,
    customer_ids: pd.Index | None = None,
    objective: str = settings.optimization_objective,
    positive_label: str = settings.positive_label,
) -> dict[str, Any]:
    """Run the full threshold optimization and business decision pipeline.

    Args:
        target_true: Ground-truth labels (encoded as 0/1) for the
            evaluation set. Reuses Milestone 7's held-out predictions
            — no model retraining happens here.
        target_proba: Predicted probability of churn for the same
            evaluation set.
        customer_ids: Identifiers for each customer, aligned with
            ``target_proba``. Defaults to a positional range index.
        objective: Optimization objective. Defaults to
            ``settings.optimization_objective``.
        positive_label: The label representing churn in the decision
            engine's output. Defaults to ``settings.positive_label``.

    Returns:
        A dictionary with the optimization result, the decision
        table, every generated figure path, and the generated report
        paths.
    """
    logger.info("Threshold optimization started.")
    logger.info("Prediction probabilities loaded.")

    optimization_result = optimize_threshold(target_true, target_proba, objective=objective)
    optimal_threshold = optimization_result["optimal_threshold"]
    logger.info("Business cost analysis completed.")

    probabilities = pd.Series(target_proba)
    decision_table = generate_recommendations(
        probabilities, optimal_threshold, customer_ids=customer_ids, positive_label=positive_label
    )
    logger.info("Decision engine executed.")

    evaluation_table = optimization_result["evaluation_table"]
    figure_paths: dict[str, Any] = {
        **plot_threshold_metrics(evaluation_table, optimal_threshold),
        "threshold_vs_cost": plot_business_cost_curve(evaluation_table, optimal_threshold),
        "precision_recall_tradeoff": plot_precision_recall_tradeoff(
            evaluation_table, optimal_threshold
        ),
        "optimal_threshold_confusion_matrix": plot_optimal_confusion_matrix(
            target_true, target_proba, optimal_threshold
        ),
    }

    save_threshold_config(
        optimal_threshold, objective, optimization_result["metrics_at_optimal_threshold"]
    )

    report_paths = generate_threshold_report(optimization_result, decision_table, figure_paths)
    logger.info("Reports generated.")
    logger.info("Optimization completed.")

    return {
        "optimization_result": optimization_result,
        "decision_table": decision_table,
        "figure_paths": figure_paths,
        "report_paths": report_paths,
    }
