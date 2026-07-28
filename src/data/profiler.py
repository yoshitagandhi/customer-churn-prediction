"""Dataset profiling and metadata generation.

This module describes a dataset — feature types, cardinality, memory
usage, target distribution — without judging whether the data is
"correct". Validation concerns belong in :mod:`src.data.validator`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)

_BINARY_CARDINALITY = 2
_BYTES_PER_MEGABYTE = 1024**2


def profile_dataset(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
) -> dict[str, Any]:
    """Generate descriptive metadata for a dataset.

    Args:
        dataframe:
            Dataset to profile.

        target_column:
            Name of the target column.

    Returns:
        Dictionary containing dataset metadata.

    Raises:
        DataValidationError:
            If the dataframe is empty.
    """
    if dataframe.empty:
        raise DataValidationError("Cannot profile an empty dataset.")

    logger.info(
        "Profiling dataset (%d rows, %d columns).",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    numeric_features = dataframe.select_dtypes(
        include=["number"],
    ).columns.tolist()

    categorical_features = dataframe.select_dtypes(
        include=["object", "category"],
    ).columns.tolist()

    binary_features = _identify_binary_features(dataframe)

    cardinality = {
        column: int(dataframe[column].nunique(dropna=True))
        for column in dataframe.columns
    }

    missing_values = {
        column: int(dataframe[column].isna().sum())
        for column in dataframe.columns
    }

    duplicate_rows = int(dataframe.duplicated().sum())

    memory_usage_bytes = int(
        dataframe.memory_usage(deep=True).sum()
    )

    profile = {
        "total_rows": int(len(dataframe)),
        "total_columns": int(len(dataframe.columns)),
        "feature_names": dataframe.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "cardinality": cardinality,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "memory_usage_bytes": memory_usage_bytes,
        "memory_usage_mb": round(
            memory_usage_bytes / _BYTES_PER_MEGABYTE,
            4,
        ),
        "target_column": target_column,
        "target_distribution": _compute_target_distribution(
            dataframe,
            target_column,
        ),
    }

    logger.info(
        "Profiling completed successfully."
    )

    logger.debug(
        "Numeric=%d | Categorical=%d | Binary=%d",
        len(numeric_features),
        len(categorical_features),
        len(binary_features),
    )

    return profile


def _identify_binary_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return columns containing exactly two unique values."""

    return [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True)
        == _BINARY_CARDINALITY
    ]


def _compute_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, dict[str, float]] | None:
    """Compute class distribution for the target column."""

    if target_column not in dataframe.columns:
        logger.debug(
            "Target column '%s' not found.",
            target_column,
        )
        return None

    value_counts = dataframe[target_column].value_counts(
        dropna=False,
    )

    total = int(value_counts.sum())

    distribution = {
        str(label): {
            "count": int(count),
            "percentage": round(
                count / total * 100,
                2,
            ),
        }
        for label, count in value_counts.items()
    }

    return distribution