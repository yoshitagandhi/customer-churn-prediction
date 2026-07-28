"""Model comparison and ranking.

Builds the evaluation comparison table and identifies the best model
using a multi-criteria ranking (ROC-AUC primary, with PR-AUC, F1, and
Recall as tie-breakers) — never a hardcoded winner.
"""

from typing import Any

import pandas as pd

from configs.logging_config import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)

# Ranking order: primary metric first, then tie-breakers, all
# descending (higher is better for every one of these metrics).
_RANKING_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "f1", "recall")


def compare_models(
    metrics_by_model: dict[str, dict[str, Any]],
    sampling_strategy_by_model: dict[str, str],
    training_time_by_model: dict[str, float],
) -> pd.DataFrame:
    """Build a comparison table across every evaluated model.

    Args:
        metrics_by_model: Mapping of model name to its metrics
            dictionary, as returned by
            :func:`src.evaluation.metrics.compute_metrics`.
        sampling_strategy_by_model: Mapping of model name to the
            sampling strategy it was trained with.
        training_time_by_model: Mapping of model name to its training
            duration in seconds (sourced from Milestone 6's experiment
            log — this milestone does not retrain anything).

    Returns:
        A DataFrame with one row per model, ranked by
        :data:`_RANKING_METRICS` (all descending).
    """
    rows: list[dict[str, Any]] = []
    for model_name, metrics in metrics_by_model.items():
        row = {
            "model_name": model_name,
            "sampling_strategy": sampling_strategy_by_model.get(model_name, "unknown"),
            "training_time_seconds": training_time_by_model.get(model_name, float("nan")),
        }
        row.update(metrics)
        rows.append(row)

    comparison_frame = pd.DataFrame(rows)
    ranking_columns = [col for col in _RANKING_METRICS if col in comparison_frame.columns]
    if ranking_columns:
        comparison_frame = comparison_frame.sort_values(
            ranking_columns, ascending=[False] * len(ranking_columns)
        ).reset_index(drop=True)

    logger.debug("Built evaluation comparison table for %d model(s).", len(comparison_frame))
    return comparison_frame


def identify_best_model(comparison_frame: pd.DataFrame) -> dict[str, Any]:
    """Identify the best-performing model from a ranked comparison table.

    Args:
        comparison_frame: Output of :func:`compare_models`, already
            sorted by the ranking criteria.

    Returns:
        A dictionary with the winning model's row data plus a
        "selection_reason" string explaining why it ranked first.

    Raises:
        ConfigurationError: If the comparison table is empty.
    """
    if comparison_frame.empty:
        raise ConfigurationError("Cannot identify best model: comparison table is empty.")

    best_row = comparison_frame.iloc[0].to_dict()
    reason = (
        f"Ranked first by ROC-AUC ({best_row.get('roc_auc')}), "
        f"with PR-AUC={best_row.get('pr_auc')}, F1={best_row.get('f1')}, "
        f"Recall={best_row.get('recall')} used as tie-breakers."
    )
    best_row["selection_reason"] = reason
    logger.info("Best model identified: '%s'.", best_row["model_name"])
    return best_row
