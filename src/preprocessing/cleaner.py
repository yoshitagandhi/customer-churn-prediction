"""Dataset cleaning.

This module performs only lightweight, reversible-in-spirit cleaning
operations required before feature engineering and encoding — it does
not encode, scale, or engineer features.

Cleaning is split into two layers:

- :func:`apply_row_level_cleaning` never changes the number of rows,
  so it is safe to reuse at inference time on a single record (via
  :mod:`src.preprocessing.pipeline`).
- :func:`clean_dataset` additionally removes exact duplicate rows,
  which only makes sense when preparing a full historical dataset —
  never at inference time on a single record.
"""

import pandas as pd

from configs.logging_config import get_logger
from src.data.schema import CUSTOMER_ID_COLUMN, NUMERIC_COLUMNS
from src.utils.helpers import coerce_numeric

logger = get_logger(__name__)


def apply_row_level_cleaning(
    dataframe: pd.DataFrame,
    id_column: str = CUSTOMER_ID_COLUMN,
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """Apply cleaning operations that never change the number of rows.

    Safe to use both on a full historical dataset and on a single
    inference record, since row count is always preserved.

    Steps:
        1. Strip whitespace from column names.
        2. Strip whitespace from string/object column values.
        3. Coerce nominally-numeric columns (e.g., "TotalCharges")
           that were read as text back to numeric, turning
           unparseable values (e.g., blank strings) into NaN.
        4. Drop the identifier column, if present.

    Args:
        dataframe: The dataset to clean.
        id_column: Name of the identifier column to drop. Defaults to
            the schema's customer ID column.
        numeric_columns: Columns that should hold numeric data.
            Defaults to the schema's numeric columns.

    Returns:
        A new, cleaned DataFrame. The input DataFrame is never
        modified.
    """
    cleaned = dataframe.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    cleaned = _trim_string_values(cleaned)
    cleaned = _coerce_numeric_columns(cleaned, numeric_columns)

    if id_column in cleaned.columns:
        cleaned = cleaned.drop(columns=[id_column])
        logger.debug("Dropped identifier column '%s'.", id_column)

    return cleaned


def clean_dataset(
    dataframe: pd.DataFrame,
    id_column: str = CUSTOMER_ID_COLUMN,
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS,
    drop_duplicate_rows: bool = True,
) -> pd.DataFrame:
    """Clean a full historical dataset before it enters the ML pipeline.

    Applies :func:`apply_row_level_cleaning`, then optionally removes
    exact duplicate rows. This function is intended to be run once on
    the full training corpus — not on individual inference records,
    since duplicate-row removal is only meaningful across a batch of
    rows.

    Args:
        dataframe: The dataset to clean.
        id_column: Name of the identifier column to drop. Defaults to
            the schema's customer ID column.
        numeric_columns: Columns that should hold numeric data.
            Defaults to the schema's numeric columns.
        drop_duplicate_rows: Whether to remove exact duplicate rows.
            Defaults to True, since duplicate rows would otherwise
            risk leaking identical records across a future train/test
            split.

    Returns:
        A new, cleaned DataFrame. The input DataFrame is never
        modified.
    """
    logger.info("Cleaning started.")
    cleaned = apply_row_level_cleaning(dataframe, id_column, numeric_columns)

    if drop_duplicate_rows:
        rows_before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        rows_removed = rows_before - len(cleaned)
        if rows_removed > 0:
            logger.warning("Removed %d exact duplicate row(s) during cleaning.", rows_removed)

    logger.info("Cleaning completed: %d rows, %d columns.", cleaned.shape[0], cleaned.shape[1])
    return cleaned


def _trim_string_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every object-dtype column.

    Args:
        dataframe: The dataset to clean.

    Returns:
        A new DataFrame with whitespace trimmed from string values.
    """
    result = dataframe.copy()
    object_columns = result.select_dtypes(include="object").columns
    for column in object_columns:
        result[column] = result[column].str.strip()
    return result


def _coerce_numeric_columns(
    dataframe: pd.DataFrame, numeric_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Coerce nominally-numeric columns stored as text back to numeric.

    Args:
        dataframe: The dataset to clean.
        numeric_columns: Columns expected to hold numeric data.

    Returns:
        A new DataFrame with the specified columns coerced to numeric
        where necessary.
    """
    result = dataframe.copy()
    for column in numeric_columns:
        if column in result.columns:
            result[column] = coerce_numeric(result[column])
    return result
