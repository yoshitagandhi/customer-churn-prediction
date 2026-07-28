"""Input validation for the Streamlit application.

Pure Python — no Streamlit imports here, so this module is testable
and reusable independent of the UI layer. Reuses the dataset schema
from Milestone 2 as the single source of truth for valid categorical
values and required columns, rather than duplicating that list.
"""

from typing import Any, Final

import pandas as pd

from src.data.schema import CUSTOMER_ID_COLUMN, REQUIRED_COLUMNS, VALID_TARGET_LABELS
from src.utils.constants import TARGET_COLUMN

# Known valid values for each categorical input field, matching the
# Telco Customer Churn dataset's documented categories (see
# src/data/schema.py for the column-level schema this complements).
VALID_CATEGORICAL_VALUES: Final[dict[str, tuple[str, ...]]] = {
    "gender": ("Female", "Male"),
    "Partner": ("Yes", "No"),
    "Dependents": ("Yes", "No"),
    "PhoneService": ("Yes", "No"),
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("Yes", "No", "No internet service"),
    "OnlineBackup": ("Yes", "No", "No internet service"),
    "DeviceProtection": ("Yes", "No", "No internet service"),
    "TechSupport": ("Yes", "No", "No internet service"),
    "StreamingTV": ("Yes", "No", "No internet service"),
    "StreamingMovies": ("Yes", "No", "No internet service"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("Yes", "No"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

# Numeric field bounds. Upper bounds are generous rather than exact,
# since real-world values can exceed the training data's observed
# range without being invalid.
NUMERIC_FIELD_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "tenure": (0, 100),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 50000),
}

REQUIRED_CUSTOMER_FIELDS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in (CUSTOMER_ID_COLUMN, TARGET_COLUMN)
)


def validate_customer_input(customer_data: dict[str, Any]) -> list[str]:
    """Validate a single customer's form input before prediction.

    Args:
        customer_data: Raw form values keyed by column name.

    Returns:
        A list of human-readable error messages. Empty if the input
        is valid.
    """
    errors: list[str] = []

    for field_name in REQUIRED_CUSTOMER_FIELDS:
        if field_name not in customer_data or customer_data[field_name] in (None, ""):
            errors.append(f"'{field_name}' is required.")

    for field_name, valid_values in VALID_CATEGORICAL_VALUES.items():
        value = customer_data.get(field_name)
        if value is not None and value not in valid_values:
            errors.append(f"'{field_name}' must be one of {valid_values}, got '{value}'.")

    for field_name, (lower_bound, upper_bound) in NUMERIC_FIELD_BOUNDS.items():
        value = customer_data.get(field_name)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{field_name}' must be numeric, got '{value}'.")
            continue
        if not (lower_bound <= numeric_value <= upper_bound):
            errors.append(f"'{field_name}' must be between {lower_bound} and {upper_bound}.")

    senior_citizen = customer_data.get("SeniorCitizen")
    if senior_citizen is not None and senior_citizen not in (0, 1, True, False):
        errors.append("'SeniorCitizen' must be 0/1 (or Yes/No).")

    return errors


def validate_batch_dataframe(dataframe: pd.DataFrame) -> list[str]:
    """Validate an uploaded batch-prediction CSV before processing.

    Args:
        dataframe: The uploaded CSV, read into a DataFrame.

    Returns:
        A list of human-readable error messages. Empty if the file is
        valid enough to proceed (individual bad rows are still
        reported separately during prediction, not blocked here).
    """
    errors: list[str] = []

    if dataframe.empty:
        errors.append("The uploaded file contains no rows.")
        return errors

    missing_columns = sorted(set(REQUIRED_CUSTOMER_FIELDS) - set(dataframe.columns))
    if missing_columns:
        errors.append(f"Missing required column(s): {missing_columns}.")

    if "Churn" in dataframe.columns:
        unexpected_labels = set(dataframe["Churn"].dropna().unique()) - set(VALID_TARGET_LABELS)
        if unexpected_labels:
            errors.append(
                f"Column 'Churn' contains unexpected label(s): {sorted(unexpected_labels)}."
            )

    return errors
