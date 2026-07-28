"""Evaluation visualizations.

Reuses the shared styling and figure-saving primitives from
:mod:`src.visualization.plots` (Milestone 3) rather than duplicating
that boilerplate — every figure here follows the same look as the
EDA figures.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.visualization.plots import save_figure

logger = get_logger(__name__)


def _slugify(name: str) -> str:
    """Convert a model name to a filesystem-safe, lowercase slug.

    Args:
        name: Model name, e.g. "logistic_regression" or "XGBoost".

    Returns:
        A lowercase, underscore-separated slug.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_roc_curve(
    model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path = FIGURES_DIR,
    filename: str = "roc_curve.png",
) -> Path:
    """Plot ROC curves for every model on a single overlaid figure.

    Args:
        model_predictions: Mapping of model display name to a
            (target_true, target_proba) tuple.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, (target_true, target_proba) in model_predictions.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(target_true, target_proba)
        auc_value = np.trapezoid(true_positive_rate, false_positive_rate)
        axis.plot(
            false_positive_rate, true_positive_rate, label=f"{model_name} (AUC={auc_value:.3f})"
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random baseline")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve — Model Comparison")
    axis.legend(loc="lower right", fontsize=9)
    return save_figure(figure, filename, output_dir)


def plot_precision_recall_curve(
    model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path = FIGURES_DIR,
    filename: str = "precision_recall_curve.png",
) -> Path:
    """Plot Precision-Recall curves for every model on a single overlaid figure.

    Args:
        model_predictions: Mapping of model display name to a
            (target_true, target_proba) tuple.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, (target_true, target_proba) in model_predictions.items():
        precision, recall, _ = precision_recall_curve(target_true, target_proba)
        axis.plot(recall, precision, label=model_name)

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve — Model Comparison")
    axis.legend(loc="lower left", fontsize=9)
    return save_figure(figure, filename, output_dir)


def plot_confusion_matrix(
    target_true: np.ndarray,
    target_pred: np.ndarray,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot a confusion matrix for a single model.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_pred: Predicted labels, encoded as 0/1.
        model_name: Model name, used in the title and filename.
        output_dir: Directory to save into.

    Returns:
        The path the figure was saved to.
    """
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        target_true,
        target_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    axis.set_title(f"Confusion Matrix — {model_name}")
    filename = f"confusion_matrix_{_slugify(model_name)}.png"
    return save_figure(figure, filename, output_dir)


def plot_classification_report_heatmap(
    metrics_by_model: dict[str, dict[str, float]],
    output_dir: Path = FIGURES_DIR,
    filename: str = "classification_report_heatmap.png",
) -> Path:
    """Plot a heatmap of precision, recall, and F1 for the given model(s).

    Args:
        metrics_by_model: Mapping of model name to its metrics
            dictionary (as returned by
            :func:`src.evaluation.metrics.compute_metrics`).
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    rows = {
        model_name: {
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
        }
        for model_name, metrics in metrics_by_model.items()
    }
    heatmap_frame = pd.DataFrame(rows).T

    figure, axis = plt.subplots(figsize=(6, max(2, 0.6 * len(heatmap_frame) + 1)))
    sns.heatmap(heatmap_frame, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=axis)
    axis.set_title("Classification Report Heatmap")
    return save_figure(figure, filename, output_dir)


def plot_metric_comparison(
    comparison_frame: pd.DataFrame,
    metrics: tuple[str, ...] = ("roc_auc", "precision", "recall", "f1"),
    output_dir: Path = FIGURES_DIR,
    filename: str = "metric_comparison.png",
) -> Path:
    """Plot a grouped bar chart comparing models across several metrics.

    Args:
        comparison_frame: A DataFrame with a "model_name" column and
            one column per metric, as produced by
            :func:`src.evaluation.comparison.compare_models`.
        metrics: Which metric columns to plot. Defaults to
            ("roc_auc", "precision", "recall", "f1").
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.
    """
    available_metrics = [metric for metric in metrics if metric in comparison_frame.columns]
    plot_frame = comparison_frame.set_index("model_name")[available_metrics]

    figure, axis = plt.subplots(figsize=(8, 5))
    plot_frame.plot(kind="bar", ax=axis)
    axis.set_ylabel("Score")
    axis.set_xlabel("Model")
    axis.set_title("Metric Comparison Across Models")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=30)
    axis.legend(title="Metric", fontsize=9)
    return save_figure(figure, filename, output_dir)
