"""Threshold-related visualizations.

Reuses :func:`src.visualization.plots.save_figure` for consistent
styling with every other figure in this project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.visualization.plots import save_figure

logger = get_logger(__name__)


def _plot_metric_vs_threshold(
    evaluation_frame: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
    filename: str,
    optimal_threshold: float | None,
    output_dir: Path,
) -> Path:
    """Plot a single metric column against the evaluated threshold range.

    Args:
        evaluation_frame: Output of
            :func:`src.threshold.threshold_analysis.evaluate_thresholds`.
        metric_column: Which column to plot on the y-axis.
        ylabel: Y-axis label.
        title: Figure title.
        filename: Output filename.
        optimal_threshold: If provided, marked with a vertical
            reference line.
        output_dir: Directory to save into.

    Returns:
        The path the figure was saved to.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        evaluation_frame["threshold"], evaluation_frame[metric_column], marker="o", markersize=3
    )
    if optimal_threshold is not None:
        axis.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal threshold={optimal_threshold:.2f}",
        )
        axis.legend()
    axis.set_xlabel("Threshold")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return save_figure(figure, filename, output_dir)


def plot_threshold_metrics(
    evaluation_frame: pd.DataFrame,
    optimal_threshold: float | None = None,
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Plot threshold-vs-precision, -recall, and -F1 charts.

    Args:
        evaluation_frame: Output of
            :func:`src.threshold.threshold_analysis.evaluate_thresholds`.
        optimal_threshold: If provided, marked with a vertical
            reference line on every chart.
        output_dir: Directory to save into.

    Returns:
        A mapping of chart identifier to the path of its generated
        figure.
    """
    return {
        "threshold_vs_precision": _plot_metric_vs_threshold(
            evaluation_frame,
            "precision",
            "Precision",
            "Threshold vs Precision",
            "threshold_vs_precision.png",
            optimal_threshold,
            output_dir,
        ),
        "threshold_vs_recall": _plot_metric_vs_threshold(
            evaluation_frame,
            "recall",
            "Recall",
            "Threshold vs Recall",
            "threshold_vs_recall.png",
            optimal_threshold,
            output_dir,
        ),
        "threshold_vs_f1": _plot_metric_vs_threshold(
            evaluation_frame,
            "f1",
            "F1 Score",
            "Threshold vs F1",
            "threshold_vs_f1.png",
            optimal_threshold,
            output_dir,
        ),
    }


def plot_business_cost_curve(
    evaluation_frame: pd.DataFrame,
    optimal_threshold: float | None = None,
    output_dir: Path = FIGURES_DIR,
    filename: str = "threshold_vs_cost.png",
) -> Path:
    """Plot business cost against the evaluated threshold range.

    Args:
        evaluation_frame: Output of
            :func:`src.threshold.threshold_analysis.evaluate_thresholds`.
        optimal_threshold: If provided, marked with a vertical
            reference line.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    return _plot_metric_vs_threshold(
        evaluation_frame,
        "business_cost",
        "Business Cost",
        "Threshold vs Business Cost",
        filename,
        optimal_threshold,
        output_dir,
    )


def plot_precision_recall_tradeoff(
    evaluation_frame: pd.DataFrame,
    optimal_threshold: float | None = None,
    output_dir: Path = FIGURES_DIR,
    filename: str = "precision_recall_tradeoff.png",
) -> Path:
    """Plot precision and recall together against the threshold range.

    Args:
        evaluation_frame: Output of
            :func:`src.threshold.threshold_analysis.evaluate_thresholds`.
        optimal_threshold: If provided, marked with a vertical
            reference line.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(evaluation_frame["threshold"], evaluation_frame["precision"], label="Precision")
    axis.plot(evaluation_frame["threshold"], evaluation_frame["recall"], label="Recall")
    if optimal_threshold is not None:
        axis.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal threshold={optimal_threshold:.2f}",
        )
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_title("Precision-Recall Trade-off vs Threshold")
    axis.legend()
    return save_figure(figure, filename, output_dir)


def plot_optimal_confusion_matrix(
    target_true: np.ndarray,
    target_proba: np.ndarray,
    optimal_threshold: float,
    output_dir: Path = FIGURES_DIR,
    filename: str = "optimal_threshold_confusion_matrix.png",
) -> Path:
    """Plot the confusion matrix produced by applying the optimal threshold.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.
        optimal_threshold: The selected classification threshold.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    target_pred = (target_proba >= optimal_threshold).astype(int)
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        target_true,
        target_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    axis.set_title(f"Confusion Matrix at Optimal Threshold ({optimal_threshold:.2f})")
    return save_figure(figure, filename, output_dir)
