"""Feature distribution visualization.

This module focuses only on visualizing how individual features are
distributed — numeric histograms/boxplots and categorical count
plots. It does not compute statistics (see
:mod:`src.analysis.statistics`) and does not compare features against
the target (see :mod:`src.visualization.business_charts`).
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.data.schema import CONTINUOUS_NUMERIC_FEATURES, KEY_CATEGORICAL_FEATURES
from src.utils.helpers import coerce_numeric
from src.visualization.plots import plot_boxplot, plot_countplot, plot_histogram, save_figure

logger = get_logger(__name__)

_NUMERICAL_DISTRIBUTION_FILENAME = "numerical_distribution.png"


def analyze_numerical_distributions(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...] = CONTINUOUS_NUMERIC_FEATURES,
    output_dir: Path = FIGURES_DIR,
) -> dict[str, dict[str, Path]]:
    """Generate a histogram and boxplot for each continuous numeric feature.

    Also generates a single combined grid figure
    (``numerical_distribution.png``) showing all continuous numeric
    features side by side for a quick overview.

    Args:
        dataframe: The dataset to visualize.
        columns: Numeric columns to analyze. Defaults to
            ``CONTINUOUS_NUMERIC_FEATURES``.
        output_dir: Directory figures are saved into.

    Returns:
        A mapping of column name to its generated figure paths
        (keys: "histogram", "boxplot"), plus a "combined" key holding
        the path to the grid overview figure.
    """
    available_columns = [column for column in columns if column in dataframe.columns]
    results: dict[str, dict[str, Path]] = {}

    for column in available_columns:
        series = coerce_numeric(dataframe[column])
        histogram_path = plot_histogram(
            series,
            title=f"Distribution of {column}",
            filename=f"{_to_snake_case(column)}_distribution.png",
            output_dir=output_dir,
        )
        boxplot_frame = dataframe.assign(**{column: series})
        boxplot_path = plot_boxplot(
            boxplot_frame,
            numeric_column=column,
            title=f"Boxplot of {column}",
            filename=f"{_to_snake_case(column)}_boxplot.png",
            output_dir=output_dir,
        )
        results[column] = {"histogram": histogram_path, "boxplot": boxplot_path}

    if available_columns:
        results["combined"] = {
            "grid": _plot_combined_numerical_grid(dataframe, available_columns, output_dir)
        }

    logger.info(
        "Generated numerical distribution figures for %d feature(s).", len(available_columns)
    )
    return results


def analyze_categorical_distributions(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...] = KEY_CATEGORICAL_FEATURES,
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Generate a count plot (with percentage labels) for each categorical feature.

    Args:
        dataframe: The dataset to visualize.
        columns: Categorical columns to analyze. Defaults to
            ``KEY_CATEGORICAL_FEATURES``.
        output_dir: Directory figures are saved into.

    Returns:
        A mapping of column name to the path of its generated figure.
    """
    results: dict[str, Path] = {}
    for column in columns:
        if column not in dataframe.columns:
            continue
        results[column] = plot_countplot(
            dataframe,
            category_column=column,
            title=f"Distribution of {column}",
            filename=f"{_to_snake_case(column)}_distribution.png",
            output_dir=output_dir,
        )

    logger.info("Generated categorical distribution figures for %d feature(s).", len(results))
    return results


def _plot_combined_numerical_grid(
    dataframe: pd.DataFrame, columns: list[str], output_dir: Path = FIGURES_DIR
) -> Path:
    """Plot a single grid figure with a histogram per continuous numeric feature.

    Args:
        dataframe: The dataset to visualize.
        columns: Numeric columns to include in the grid.
        output_dir: Directory figures are saved into.

    Returns:
        The path the combined grid figure was saved to.
    """
    figure, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4.5))
    axes = [axes] if len(columns) == 1 else axes

    for axis, column in zip(axes, columns, strict=True):
        series = coerce_numeric(dataframe[column]).dropna()
        sns.histplot(series, kde=True, ax=axis)
        axis.set_title(column)

    figure.suptitle("Numerical Feature Distributions", fontweight="bold")
    return save_figure(figure, _NUMERICAL_DISTRIBUTION_FILENAME, output_dir)


def _to_snake_case(column_name: str) -> str:
    """Convert a PascalCase/camelCase column name to snake_case for filenames.

    Handles acronym sequences correctly (e.g., "StreamingTV" becomes
    "streaming_tv", not "streaming_t_v").

    Args:
        column_name: Column name such as "MonthlyCharges" or "StreamingTV".

    Returns:
        A lowercase, underscore-separated string such as
        "monthly_charges" or "streaming_tv".
    """
    step_one = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", column_name)
    step_two = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step_one)
    return step_two.lower()
