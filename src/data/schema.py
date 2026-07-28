"""Expected schema definition for the Telco Customer Churn dataset.

This module defines the expected structure of the raw dataset used
throughout the project. It contains no validation logic; instead,
other modules reference these definitions when validating or
processing data.

Keeping the schema centralized avoids duplicated column names and
ensures that future dataset revisions require changes in only one
location.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.utils.constants import TARGET_COLUMN


class ColumnCategory(str, Enum):
    """High-level category describing a dataset column."""

    IDENTIFIER = "identifier"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


CUSTOMER_ID_COLUMN: Final[str] = "customerID"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Expected specification for a dataset column."""

    name: str
    category: ColumnCategory
    required: bool = True


EXPECTED_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec(CUSTOMER_ID_COLUMN, ColumnCategory.IDENTIFIER),
    ColumnSpec("gender", ColumnCategory.CATEGORICAL),
    ColumnSpec("SeniorCitizen", ColumnCategory.NUMERIC),
    ColumnSpec("Partner", ColumnCategory.CATEGORICAL),
    ColumnSpec("Dependents", ColumnCategory.CATEGORICAL),
    ColumnSpec("tenure", ColumnCategory.NUMERIC),
    ColumnSpec("PhoneService", ColumnCategory.CATEGORICAL),
    ColumnSpec("MultipleLines", ColumnCategory.CATEGORICAL),
    ColumnSpec("InternetService", ColumnCategory.CATEGORICAL),
    ColumnSpec("OnlineSecurity", ColumnCategory.CATEGORICAL),
    ColumnSpec("OnlineBackup", ColumnCategory.CATEGORICAL),
    ColumnSpec("DeviceProtection", ColumnCategory.CATEGORICAL),
    ColumnSpec("TechSupport", ColumnCategory.CATEGORICAL),
    ColumnSpec("StreamingTV", ColumnCategory.CATEGORICAL),
    ColumnSpec("StreamingMovies", ColumnCategory.CATEGORICAL),
    ColumnSpec("Contract", ColumnCategory.CATEGORICAL),
    ColumnSpec("PaperlessBilling", ColumnCategory.CATEGORICAL),
    ColumnSpec("PaymentMethod", ColumnCategory.CATEGORICAL),
    ColumnSpec("MonthlyCharges", ColumnCategory.NUMERIC),
    ColumnSpec("TotalCharges", ColumnCategory.NUMERIC),
    ColumnSpec(TARGET_COLUMN, ColumnCategory.CATEGORICAL),
)

EXPECTED_COLUMN_COUNT: Final[int] = len(EXPECTED_COLUMNS)

ALL_COLUMN_NAMES: Final[tuple[str, ...]] = tuple(
    column.name
    for column in EXPECTED_COLUMNS
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(
    column.name
    for column in EXPECTED_COLUMNS
    if column.required
)

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = tuple(
    column.name
    for column in EXPECTED_COLUMNS
    if column.category == ColumnCategory.IDENTIFIER
)

NUMERIC_COLUMNS: Final[tuple[str, ...]] = tuple(
    column.name
    for column in EXPECTED_COLUMNS
    if column.category == ColumnCategory.NUMERIC
)

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    column.name
    for column in EXPECTED_COLUMNS
    if column.category == ColumnCategory.CATEGORICAL
)

COLUMN_SPECS: Final[dict[str, ColumnSpec]] = {
    column.name: column
    for column in EXPECTED_COLUMNS
}

COLUMN_CATEGORIES: Final[dict[str, ColumnCategory]] = {
    column.name: column.category
    for column in EXPECTED_COLUMNS
}

VALID_TARGET_LABELS: Final[tuple[str, ...]] = (
    "Yes",
    "No",
)

CONTINUOUS_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

KEY_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = tuple(
    column
    for column in CATEGORICAL_COLUMNS
    if column not in (
        CUSTOMER_ID_COLUMN,
        TARGET_COLUMN,
    )
)