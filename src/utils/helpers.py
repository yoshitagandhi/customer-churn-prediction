"""Generic helper functions shared across the project.

These utilities are intentionally domain-agnostic: they do not load
data, engineer features, or train models. Domain-specific logic
belongs in its corresponding ``src`` subpackage instead.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, ParamSpec

import pandas as pd

from configs.logging_config import get_logger
from src.utils.exceptions import ConfigurationError

try:
    import numpy as np
except ImportError:  #pragma: no cover
    np = None

logger = get_logger(__name__)

_ReturnType = TypeVar("_ReturnType")
_Params = ParamSpec("_Params")

def _calculate_churn_rate(
    values: pd.Series,
    positive_label: str,
) -> float:
    """Return the churn rate percentage for a grouped Series."""
    return float((values == positive_label).mean() * 100)


def compute_churn_rate_by_category(
    dataframe: pd.DataFrame,
    category_column: str,
    target_column: str,
    positive_label: str = "Yes",
) -> dict[str, float]:
    """Compute the churn rate (%) for each category of a given column.

    Args:
        dataframe:
            Dataset containing the categorical feature and target.
            
        category_column:
            Name of the categorical column.
            
        target_column:
            Name of the churn target column.
            
        positive_label:
            Label representing churn.
            Defaults to ``"Yes"``.

    Returns:
        Dictionary mapping each category to its churn rate percentage,
        sorted from highest to lowest.
        
    Raises:
        ValueError:
            If the dataframe is empty or required columns are missing.
    """
    
    if dataframe.empty:
        raise ValueError("Cannot compute churn rates from an empty dataframe.")
    
    required_columns = {category_column, target_column}
    missing_columns = required_columns.difference(dataframe.columns)
    
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    grouped = dataframe.groupby(
        category_column, 
        dropna=False, 
        observed=True
        )[target_column]
    
    churn_rate = grouped.apply(
        _calculate_churn_rate,
        positive_label=positive_label,
        )
    
    sorted_rates = churn_rate.sort_values(ascending=False)
    
    result = {
        str(category): round(float(rate), 2)
        for category, rate in sorted_rates.items()
    }
    
    return result

def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, without mutating the caller's data.

    Non-numeric values are coerced to ``NaN`` while preserving the
    original Series.

    Args:
        series:
            The Series to convert.

    Returns:
        Numeric Series.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    coerced = pd.to_numeric(series, errors="coerce")
    
    original_missing = int(series.isna().sum())
    coerced_missing = int(coerced.isna().sum())
    newly_missing = int(coerced.isna().sum() - series.isna().sum())
    
    if newly_missing > 0:
        logger.warning(
            "%d non-numeric value(s) in '%s' coerced to NaN.",
            newly_missing, 
            series.name
        )
    
    return coerced


def set_global_random_seed(seed: int) -> None:
    """Set the random seed for all supported libraries for reproducibility.

    Currently seeds Python's built-in random module and NumPy (if
    installed).

    Args:
        seed: The seed value to apply.

    Returns:
        None.
    """
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)
    else:
        logger.debug("NumPy not installed; skipping NumPy seed.")

    logger.debug("Global random seed set to %s.", seed)


def verify_path_exists(
    path: Path, 
    *, 
    description: str = "Path",
    must_be_file: bool |None = None,
) -> Path:
    """Verify that a filesystem path exists, raising a clear error if not.

    Args:
        path: The filesystem path to check.
        
        description: A human-readable label used in the error message
            to identify what the path represents (e.g., "Dataset file").
            
        must_be_file:
            True: path must be a file.
            False: path must be a directory.
            None: only existence is checked.


    Returns:
        The validated path.
        
    Raises:
        ConfigurationError: 
            If validation falls.
    """
    if not path.exists():
        raise ConfigurationError(f"{description} does not exist: {path}")
    
    if must_be_file is True and not path.is_file():
        raise ConfigurationError(f"{description} must be a file: {path}")

    if must_be_file is False and not path.is_dir():
        raise ConfigurationError(f"{description} must be a directory: {path}")
    
    return path


def timer(
    func: Callable[_Params, _ReturnType],
) -> Callable[_Params, _ReturnType]:
    """Log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(
        *args: _Params.args,
        **kwargs: _Params.kwargs,
    ) -> _ReturnType:
        start_time = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time

            logger.debug(
                "%s completed in %.4f seconds.",
                func.__name__,
                elapsed,
            )

    return wrapper
