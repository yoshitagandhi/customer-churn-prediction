"""SHAP explanation computation and visualization.

Provides reusable methods for global and local SHAP explanations that
other modules (feature importance, prediction explanation, reporting)
consume. Every SHAP explainer backend can return values shaped
differently (binary classifiers sometimes return a value per class);
:func:`extract_positive_class_values` normalizes this once so the
rest of the codebase never has to worry about it.

This module does not generate reports — only explanations and
figures.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.visualization.plots import configure_plot_style

logger = get_logger(__name__)


def extract_positive_class_values(shap_values: Any) -> np.ndarray:
    """Normalize a SHAP Explanation's values to a 2D positive-class array.

    Some SHAP explainer backends return one value per class
    (shape: rows x features x classes); others return a single value
    per feature already (shape: rows x features). This always returns
    the latter, using the positive (churn) class when a per-class
    axis is present.

    Args:
        shap_values: A SHAP Explanation object.

    Returns:
        A 2D NumPy array of shape (n_rows, n_features).
    """
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def extract_base_value(shap_values: Any) -> float:
    """Extract the positive-class expected value (SHAP's baseline) as a float.

    Args:
        shap_values: A SHAP Explanation object.

    Returns:
        The baseline (expected) prediction the SHAP contributions are
        added to.
    """
    base = np.asarray(shap_values.base_values)
    if base.ndim == 2:
        return float(base[0, -1])
    if base.ndim == 1:
        return float(base[0])
    return float(base)


def generate_global_explanations(shap_values: Any, feature_names: list[str]) -> dict[str, Any]:
    """Compute global explanations: mean |SHAP| per feature and a ranking.

    Args:
        shap_values: A SHAP Explanation object covering many rows.
        feature_names: Feature names, in the same column order as
            ``shap_values``.

    Returns:
        A dictionary with the feature ranking (highest mean |SHAP|
        first) and the mean absolute SHAP value per feature.
    """
    values = extract_positive_class_values(shap_values)
    mean_abs_shap = np.abs(values).mean(axis=0)

    ranking = sorted(zip(feature_names, mean_abs_shap, strict=True), key=lambda pair: -pair[1])
    logger.debug("Global SHAP explanation computed for %d feature(s).", len(feature_names))

    return {
        "feature_ranking": [name for name, _ in ranking],
        "mean_abs_shap": {name: round(float(value), 6) for name, value in ranking},
    }


def generate_local_explanation(
    shap_values: Any, feature_names: list[str], row_index: int
) -> dict[str, Any]:
    """Compute a single row's SHAP contributions, ranked by magnitude.

    Args:
        shap_values: A SHAP Explanation object.
        feature_names: Feature names, in the same column order as
            ``shap_values``.
        row_index: Which row (customer) to explain.

    Returns:
        A dictionary with the row's ranked feature contributions and
        the baseline value they were added to.
    """
    values = extract_positive_class_values(shap_values)[row_index]
    contributions = sorted(
        zip(feature_names, values, strict=True), key=lambda pair: abs(pair[1]), reverse=True
    )
    return {
        "row_index": row_index,
        "base_value": extract_base_value(shap_values),
        "contributions": [
            {"feature": name, "shap_value": round(float(value), 6)} for name, value in contributions
        ],
    }


def _save_current_figure(filename: str, output_dir: Path) -> Path:
    """Save the current Matplotlib figure (as left behind by a SHAP plot call).

    SHAP's plotting functions draw directly onto the current
    Matplotlib figure rather than returning a Figure object, so this
    captures whatever is currently active instead of using
    :func:`src.visualization.plots.save_figure`.

    Args:
        filename: Output filename.
        output_dir: Directory to save into.

    Returns:
        The path the figure was saved to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.debug("Saved SHAP figure: %s", output_path)
    return output_path


def plot_shap_bar(
    shap_values: Any, output_dir: Path = FIGURES_DIR, filename: str = "shap_bar.png"
) -> Path:
    """Plot the global mean |SHAP| feature-importance bar chart.

    Args:
        shap_values: A SHAP Explanation object covering many rows.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    import shap

    configure_plot_style()
    shap.plots.bar(shap_values, show=False)
    return _save_current_figure(filename, output_dir)


def plot_shap_beeswarm(
    shap_values: Any, output_dir: Path = FIGURES_DIR, filename: str = "shap_beeswarm.png"
) -> Path:
    """Plot the SHAP beeswarm plot (per-sample feature impact distribution).

    Args:
        shap_values: A SHAP Explanation object covering many rows.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    import shap

    configure_plot_style()
    shap.plots.beeswarm(shap_values, show=False)
    return _save_current_figure(filename, output_dir)


def plot_shap_summary(
    shap_values_array: np.ndarray,
    features: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    filename: str = "shap_summary.png",
) -> Path:
    """Plot SHAP's classic summary plot.

    Args:
        shap_values_array: A 2D array of SHAP values (see
            :func:`extract_positive_class_values`).
        features: The processed feature DataFrame the values
            correspond to.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    import shap

    configure_plot_style()
    shap.summary_plot(shap_values_array, features, show=False)
    return _save_current_figure(filename, output_dir)


def plot_shap_dependence(
    feature_name: str,
    shap_values_array: np.ndarray,
    features: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    filename: str = "shap_dependence.png",
) -> Path:
    """Plot a SHAP dependence plot for a single feature.

    Shows how a feature's own value relates to its SHAP contribution,
    revealing non-linear effects and interactions.

    Args:
        feature_name: Which feature to plot (typically the top-ranked
            feature from :func:`generate_global_explanations`).
        shap_values_array: A 2D array of SHAP values (see
            :func:`extract_positive_class_values`).
        features: The processed feature DataFrame the values
            correspond to.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    import shap

    configure_plot_style()
    shap.dependence_plot(feature_name, shap_values_array, features, show=False)
    return _save_current_figure(filename, output_dir)


def plot_shap_waterfall(
    shap_values: Any,
    row_index: int,
    output_dir: Path = FIGURES_DIR,
    filename: str = "shap_waterfall.png",
) -> Path:
    """Plot a SHAP waterfall plot explaining a single prediction.

    Args:
        shap_values: A SHAP Explanation object covering many rows.
        row_index: Which row (customer) to plot.
        output_dir: Directory to save into.
        filename: Output filename.

    Returns:
        The path the figure was saved to.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    import shap

    configure_plot_style()
    single_explanation = shap_values[row_index]
    if np.asarray(single_explanation.values).ndim == 2:
        single_explanation = single_explanation[:, -1]
    shap.plots.waterfall(single_explanation, show=False)
    return _save_current_figure(filename, output_dir)
