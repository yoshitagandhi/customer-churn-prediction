"""Descriptive statistics generation for exploratory data analysis.

This module computes statistics only — it does not visualize, does
not generate insights, and does not modify the input DataFrame. Any
numeric coercion performed here (e.g., for a nominally-numeric column
stored as text) is applied to a local copy purely so a statistic can
be computed; the caller's DataFrame is never mutated.
"""

from typing import Any, Final

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.data.schema import CONTINUOUS_NUMERIC_FEATURES, KEY_CATEGORICAL_FEATURES
from src.utils.helpers import coerce_numeric
from src.utils.helpers import compute_churn_rate_by_category as _compute_churn_rate_by_category

logger = get_logger(__name__)

# Re-exported for backward compatibility with modules that imported
# these feature groupings from here before they were centralized in
# ``src.data.schema`` (the single source of truth shared by both
# ``src.analysis`` and ``src.visualization``, which would otherwise
# form a circular import).
__all_reexports__ = ("CONTINUOUS_NUMERIC_FEATURES", "KEY_CATEGORICAL_FEATURES")


def compute_dataset_statistics(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level dataset statistics.

    Args:
        dataframe: The dataset to summarize.

    Returns:
        A dictionary with row count, column count, and memory usage.
    """
    memory_usage_bytes = int(dataframe.memory_usage(deep=True).sum())
    return {
        "total_rows": int(dataframe.shape[0]),
        "total_columns": int(dataframe.shape[1]),
        "memory_usage_mb": round(memory_usage_bytes / (1024**2), 4),
    }


def compute_numerical_statistics(
    dataframe: pd.DataFrame, columns: tuple[str, ...] = CONTINUOUS_NUMERIC_FEATURES
) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics for numerical columns.

    Args:
        dataframe: The dataset to analyze.
        columns: Numeric columns to summarize. Defaults to
            ``CONTINUOUS_NUMERIC_FEATURES``.

    Returns:
        A mapping of column name to its statistics: mean, median,
        std, variance, min, max, 25th/75th percentiles, skewness, and
        kurtosis.
    """
    statistics: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in dataframe.columns:
            logger.debug("Column '%s' not found; skipping numerical statistics.", column)
            continue

        series = coerce_numeric(dataframe[column]).dropna()
        if series.empty:
            logger.warning("Column '%s' has no valid numeric values; skipping.", column)
            continue

        statistics[column] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "variance": round(float(series.var()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(float(series.quantile(0.25)), 4),
            "q3": round(float(series.quantile(0.75)), 4),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurt()), 4),
        }
    return statistics


def compute_categorical_statistics(
    dataframe: pd.DataFrame, columns: tuple[str, ...] = KEY_CATEGORICAL_FEATURES
) -> dict[str, dict[str, Any]]:
    """Compute descriptive statistics for categorical columns.

    Args:
        dataframe: The dataset to analyze.
        columns: Categorical columns to summarize. Defaults to
            ``KEY_CATEGORICAL_FEATURES``.

    Returns:
        A mapping of column name to its cardinality, mode, and
        frequency table (count and percentage per category).
    """
    statistics: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in dataframe.columns:
            logger.debug("Column '%s' not found; skipping categorical statistics.", column)
            continue

        value_counts = dataframe[column].value_counts(dropna=False)
        total = len(dataframe)
        frequency_table = {
            str(label): {
                "count": int(count),
                "percentage": round(float(count) / total * 100, 2),
            }
            for label, count in value_counts.items()
        }
        column_mode = dataframe[column].mode(dropna=True)
        statistics[column] = {
            "cardinality": int(dataframe[column].nunique(dropna=True)),
            "mode": str(column_mode.iloc[0]) if not column_mode.empty else None,
            "frequency_table": frequency_table,
        }
    return statistics


def compute_target_statistics(
    dataframe: pd.DataFrame, target_column: str = settings.target_column
) -> dict[str, Any]:
    """Compute the target variable's class distribution and imbalance ratio.

    Args:
        dataframe: The dataset to analyze.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.

    Returns:
        A dictionary with per-class counts/percentages and the
        imbalance ratio (majority class count / minority class count).
    """
    value_counts = dataframe[target_column].value_counts(dropna=False)
    total = len(dataframe)
    distribution = {
        str(label): {
            "count": int(count),
            "percentage": round(float(count) / total * 100, 2),
        }
        for label, count in value_counts.items()
    }
    imbalance_ratio = round(float(value_counts.max()) / float(value_counts.min()), 2)
    return {
        "target_column": target_column,
        "distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
    }


def compute_correlation_matrix(
    dataframe: pd.DataFrame, columns: tuple[str, ...] = CONTINUOUS_NUMERIC_FEATURES
) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for numeric columns.

    Args:
        dataframe: The dataset to analyze.
        columns: Numeric columns to include. Defaults to
            ``CONTINUOUS_NUMERIC_FEATURES``.

    Returns:
        A correlation matrix DataFrame indexed and columned by the
        available numeric columns.
    """
    available_columns = [column for column in columns if column in dataframe.columns]
    numeric_frame = pd.DataFrame(
        {column: coerce_numeric(dataframe[column]) for column in available_columns}
    )
    return numeric_frame.corr()


def get_top_correlations(
    correlation_matrix: pd.DataFrame, top_n: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Identify the strongest positive and negative feature correlations.

    Args:
        correlation_matrix: A square correlation matrix, as returned
            by ``compute_correlation_matrix``.
        top_n: How many strongest correlations to return per direction.

    Returns:
        A dictionary with "strongest_positive" and "strongest_negative"
        lists, each containing dicts of the two feature names and
        their correlation coefficient.
    """
    pairs: list[dict[str, Any]] = []
    columns = correlation_matrix.columns
    for i, first_column in enumerate(columns):
        for second_column in columns[i + 1 :]:
            correlation_value = correlation_matrix.loc[first_column, second_column]
            if pd.isna(correlation_value):
                continue
            pairs.append(
                {
                    "feature_a": first_column,
                    "feature_b": second_column,
                    "correlation": round(float(correlation_value), 4),
                }
            )

    positive_pairs = sorted(
        (p for p in pairs if p["correlation"] > 0), key=lambda p: -p["correlation"]
    )
    negative_pairs = sorted(
        (p for p in pairs if p["correlation"] < 0), key=lambda p: p["correlation"]
    )

    return {
        "strongest_positive": positive_pairs[:top_n],
        "strongest_negative": negative_pairs[:top_n],
    }


def compute_churn_rate_by_category(
    dataframe: pd.DataFrame,
    category_column: str,
    target_column: str = settings.target_column,
    positive_label: str = "Yes",
) -> dict[str, float]:
    """Compute the churn rate (%) for each category of a given column.

    Thin wrapper around :func:`src.utils.helpers.compute_churn_rate_by_category`
    that defaults ``target_column`` to ``settings.target_column`` for
    convenience within this module.

    Args:
        dataframe: The dataset to analyze.
        category_column: Categorical column to group by.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        positive_label: The target label that represents churn.
            Defaults to "Yes".

    Returns:
        A mapping of category value to its churn rate percentage,
        sorted from highest to lowest churn rate.
    """
    return _compute_churn_rate_by_category(
        dataframe, category_column, target_column, positive_label
    )
