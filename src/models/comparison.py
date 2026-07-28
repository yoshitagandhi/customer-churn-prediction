"""Model comparison.

Turns a list of experiment records into a single comparison table and
identifies the best-performing model. No visualization here — that
belongs to Milestone 7.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.models.experiment_tracker import ExperimentRecord
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)


def compare_models(experiment_records: list[ExperimentRecord]) -> pd.DataFrame:
    """Build a comparison table across every trained model's experiment record.

    Args:
        experiment_records: Records produced by
            :func:`src.models.experiment_tracker.log_experiment`.

    Returns:
        A DataFrame with one row per experiment, columns for the
        model name, sampling strategy, training time, and every
        validation metric, sorted by ``settings.scoring_metric``
        descending when that column is present.
    """
    rows: list[dict[str, Any]] = []
    for record in experiment_records:
        row = {
            "model_name": record.model_name,
            "sampling_strategy": record.sampling_strategy,
            "training_time_seconds": record.training_time_seconds,
        }
        row.update(record.validation_metrics)
        rows.append(row)

    comparison_frame = pd.DataFrame(rows)
    if settings.scoring_metric in comparison_frame.columns:
        comparison_frame = comparison_frame.sort_values(
            settings.scoring_metric, ascending=False
        ).reset_index(drop=True)

    logger.debug("Built comparison table for %d model(s).", len(comparison_frame))
    return comparison_frame


def identify_best_model(
    comparison_frame: pd.DataFrame, metric: str = settings.scoring_metric
) -> dict[str, Any]:
    """Identify the best-performing model from a comparison table.

    Args:
        comparison_frame: Output of :func:`compare_models`.
        metric: The metric to rank by (higher is better). Defaults to
            ``settings.scoring_metric``.

    Returns:
        The best row as a dictionary (includes "model_name").

    Raises:
        ConfigurationError: If the comparison table is empty or does
            not contain ``metric``.
    """
    if comparison_frame.empty:
        raise ConfigurationError("Cannot identify best model: comparison table is empty.")
    if metric not in comparison_frame.columns:
        raise ConfigurationError(
            f"Metric '{metric}' not found in comparison results. "
            f"Available columns: {comparison_frame.columns.tolist()}"
        )

    best_row = comparison_frame.loc[comparison_frame[metric].idxmax()]
    return best_row.to_dict()


def save_comparison_table(
    comparison_frame: pd.DataFrame, path: Path = settings.model_comparison_path
) -> Path:
    """Save the comparison table to disk as CSV.

    Args:
        comparison_frame: Output of :func:`compare_models`.
        path: Destination path. Defaults to
            ``settings.model_comparison_path``.

    Returns:
        The path the table was saved to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(path, index=False)
    logger.info("Model comparison saved to %s.", path)
    return path
