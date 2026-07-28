"""Threshold optimization.

Selects the optimal classification threshold according to a
configurable objective -- minimum business cost, maximum F1, or
maximum recall subject to a minimum precision constraint. The
objective is a configuration value, never hardcoded logic scattered
across the codebase.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.threshold.threshold_analysis import evaluate_thresholds
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)


def _select_min_cost(evaluation_frame: pd.DataFrame, **_: Any) -> pd.Series:
    """Select the threshold with the lowest business cost."""
    return evaluation_frame.loc[evaluation_frame["business_cost"].idxmin()]


def _select_max_f1(evaluation_frame: pd.DataFrame, **_: Any) -> pd.Series:
    """Select the threshold with the highest F1 score."""
    return evaluation_frame.loc[evaluation_frame["f1"].idxmax()]


def _select_max_recall_min_precision(
    evaluation_frame: pd.DataFrame,
    min_precision: float = settings.min_precision_constraint,
    **_: Any,
) -> pd.Series:
    """Select the highest-recall threshold among those meeting a minimum precision."""
    eligible = evaluation_frame[evaluation_frame["precision"] >= min_precision]
    if eligible.empty:
        logger.warning(
            "No threshold met the minimum precision constraint (%.2f); falling back to max F1.",
            min_precision,
        )
        return _select_max_f1(evaluation_frame)
    return eligible.loc[eligible["recall"].idxmax()]


_OBJECTIVE_DISPATCH = {
    "min_cost": _select_min_cost,
    "max_f1": _select_max_f1,
    "max_recall_min_precision": _select_max_recall_min_precision,
}

SUPPORTED_OBJECTIVES: tuple[str, ...] = tuple(_OBJECTIVE_DISPATCH.keys())


def optimize_threshold(
    target_true: np.ndarray,
    target_proba: np.ndarray,
    objective: str = settings.optimization_objective,
    threshold_range: tuple[float, float] = settings.threshold_range,
    threshold_step: float = settings.threshold_step,
    min_precision: float = settings.min_precision_constraint,
) -> dict[str, Any]:
    """Evaluate a range of thresholds and select the optimal one.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.
        objective: Optimization objective -- one of
            `SUPPORTED_OBJECTIVES`. Defaults to
            `settings.optimization_objective`.
        threshold_range: (min, max) thresholds to evaluate. Defaults
            to `settings.threshold_range`.
        threshold_step: Step size between evaluated thresholds.
            Defaults to `settings.threshold_step`.
        min_precision: Minimum precision required when `objective`
            is "max_recall_min_precision". Defaults to
            `settings.min_precision_constraint`.

    Returns:
        A dictionary with the selected optimal threshold, the
        objective used, the full metrics row at that threshold, and
        the complete threshold evaluation table.

    Raises:
        ConfigurationError: If `objective` is not supported.
    """
    if objective not in _OBJECTIVE_DISPATCH:
        raise ConfigurationError(
            f"Unsupported optimization objective '{objective}'. Supported: {SUPPORTED_OBJECTIVES}"
        )

    logger.info("Threshold optimization started.")
    evaluation_frame = evaluate_thresholds(
        target_true, target_proba, threshold_range, threshold_step
    )

    selected_row = _OBJECTIVE_DISPATCH[objective](evaluation_frame, min_precision=min_precision)
    optimal_threshold = float(selected_row["threshold"])
    logger.info("Optimal threshold selected: %.4f (objective='%s').", optimal_threshold, objective)

    return {
        "optimal_threshold": optimal_threshold,
        "objective": objective,
        "metrics_at_optimal_threshold": selected_row.to_dict(),
        "evaluation_table": evaluation_frame,
    }


def save_threshold_config(
    optimal_threshold: float,
    objective: str,
    metrics_at_optimal_threshold: dict[str, Any],
    path: Path = settings.threshold_config_path,
) -> Path:
    """Persist the selected threshold for reuse during inference and deployment.

    Args:
        optimal_threshold: The selected classification threshold.
        objective: The optimization objective used to select it.
        metrics_at_optimal_threshold: The metrics row at the optimal
            threshold, for traceability.
        path: Destination path. Defaults to
            `settings.threshold_config_path`.

    Returns:
        The path the threshold configuration was saved to.
    """
    config_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "optimal_threshold": optimal_threshold,
        "objective": objective,
        "metrics_at_optimal_threshold": metrics_at_optimal_threshold,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_data, indent=2, default=str), encoding="utf-8")
    logger.info("Threshold configuration saved to %s.", path)
    return path


def load_threshold_config(path: Path = settings.threshold_config_path) -> dict[str, Any]:
    """Load a previously saved threshold configuration.

    Args:
        path: Location of the threshold configuration. Defaults to
            `settings.threshold_config_path`.

    Returns:
        The threshold configuration dictionary.

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Threshold configuration not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
