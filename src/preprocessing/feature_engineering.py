"""Feature engineering.

Every function here adds new columns derived only from existing
feature columns — never from the target column — so none of them can
introduce target leakage. The original columns are always preserved
unchanged; only new columns are added.

All engineered features use fixed, deterministic rules (not
statistics computed from the current batch), so they behave
identically whether applied to the full historical dataset or to a
single inference record.
"""

from typing import Final

import numpy as np
import pandas as pd

from configs.logging_config import get_logger
from src.utils.helpers import coerce_numeric

logger = get_logger(__name__)

# Fixed tenure-group boundaries (in months). Kept as constants rather
# than derived from the current batch's min/max so a single inference
# record can be grouped identically to the training data.
_TENURE_GROUP_BIN_EDGES: Final[tuple[float, ...]] = (-1, 12, 24, 48, float("inf"))
_TENURE_GROUP_LABELS: Final[tuple[str, ...]] = (
    "New Customer",
    "Short-Term",
    "Medium-Term",
    "Long-Term",
)

# Business-domain mapping from contract length to a qualitative churn
# risk category (shorter commitment = higher risk of churn).
_CONTRACT_RISK_MAPPING: Final[dict[str, str]] = {
    "Month-to-month": "High Risk",
    "One year": "Medium Risk",
    "Two year": "Low Risk",
}

# Add-on service columns used to compute a customer's total service count.
_SERVICE_COLUMNS: Final[tuple[str, ...]] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

# Names of every column this module can add, documented here so
# `src.preprocessing.metadata` has a single source of truth for the
# engineered feature list.
ENGINEERED_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "TenureGroup",
    "AverageMonthlySpend",
    "EstimatedCLV",
    "ContractRisk",
    "ServiceCount",
)


def add_tenure_group(dataframe: pd.DataFrame, tenure_column: str = "tenure") -> pd.DataFrame:
    """Add a "TenureGroup" column bucketing tenure into fixed lifecycle stages.

    Args:
        dataframe: Input data.
        tenure_column: Name of the tenure (months) column.

    Returns:
        A new DataFrame with a "TenureGroup" column added, or the
        input unchanged (copied) if ``tenure_column`` is absent.
    """
    result = dataframe.copy()
    if tenure_column not in result.columns:
        logger.debug("Column '%s' not found; skipping TenureGroup.", tenure_column)
        return result

    tenure = coerce_numeric(result[tenure_column])
    result["TenureGroup"] = pd.cut(
        tenure, bins=_TENURE_GROUP_BIN_EDGES, labels=_TENURE_GROUP_LABELS
    ).astype(str)
    return result


def add_average_monthly_spend(
    dataframe: pd.DataFrame,
    tenure_column: str = "tenure",
    total_charges_column: str = "TotalCharges",
    monthly_charges_column: str = "MonthlyCharges",
) -> pd.DataFrame:
    """Add an "AverageMonthlySpend" column: total charges spread over tenure.

    This can differ meaningfully from the customer's *current*
    ``MonthlyCharges`` when their rate has changed over time. For
    brand-new customers (tenure of 0, with no billing history yet),
    their current monthly rate is used as the best available estimate.

    Args:
        dataframe: Input data.
        tenure_column: Name of the tenure (months) column.
        total_charges_column: Name of the total charges column.
        monthly_charges_column: Name of the monthly charges column.

    Returns:
        A new DataFrame with an "AverageMonthlySpend" column added, or
        the input unchanged (copied) if any required column is absent.
    """
    result = dataframe.copy()
    required_columns = (tenure_column, total_charges_column, monthly_charges_column)
    if not all(column in result.columns for column in required_columns):
        logger.debug("Missing required column(s); skipping AverageMonthlySpend.")
        return result

    tenure = coerce_numeric(result[tenure_column])
    total_charges = coerce_numeric(result[total_charges_column])
    monthly_charges = coerce_numeric(result[monthly_charges_column])

    average_spend = np.where(tenure > 0, total_charges / tenure.replace(0, np.nan), monthly_charges)
    result["AverageMonthlySpend"] = pd.Series(average_spend, index=result.index).round(2)
    return result


def add_estimated_clv(
    dataframe: pd.DataFrame,
    tenure_column: str = "tenure",
    monthly_charges_column: str = "MonthlyCharges",
) -> pd.DataFrame:
    """Add an "EstimatedCLV" column approximating value generated to date.

    Computed as ``MonthlyCharges * tenure``: a simple, transparent
    proxy for revenue generated so far, using only variables already
    present in the dataset (no external assumptions about future
    retention or industry benchmarks). Unlike ``TotalCharges``, this
    recomputed value is always available even when ``TotalCharges``
    itself is missing (e.g., for brand-new customers).

    Args:
        dataframe: Input data.
        tenure_column: Name of the tenure (months) column.
        monthly_charges_column: Name of the monthly charges column.

    Returns:
        A new DataFrame with an "EstimatedCLV" column added, or the
        input unchanged (copied) if a required column is absent.
    """
    result = dataframe.copy()
    required_columns = (tenure_column, monthly_charges_column)
    if not all(column in result.columns for column in required_columns):
        logger.debug("Missing required column(s); skipping EstimatedCLV.")
        return result

    tenure = coerce_numeric(result[tenure_column])
    monthly_charges = coerce_numeric(result[monthly_charges_column])
    result["EstimatedCLV"] = (tenure * monthly_charges).round(2)
    return result


def add_contract_risk(dataframe: pd.DataFrame, contract_column: str = "Contract") -> pd.DataFrame:
    """Add a "ContractRisk" column mapping contract length to a risk category.

    Args:
        dataframe: Input data.
        contract_column: Name of the contract-type column.

    Returns:
        A new DataFrame with a "ContractRisk" column added, or the
        input unchanged (copied) if ``contract_column`` is absent.
    """
    result = dataframe.copy()
    if contract_column not in result.columns:
        logger.debug("Column '%s' not found; skipping ContractRisk.", contract_column)
        return result

    result["ContractRisk"] = result[contract_column].map(_CONTRACT_RISK_MAPPING).fillna("Unknown")
    return result


def add_service_count(
    dataframe: pd.DataFrame, service_columns: tuple[str, ...] = _SERVICE_COLUMNS
) -> pd.DataFrame:
    """Add a "ServiceCount" column: number of add-on services a customer subscribes to.

    Args:
        dataframe: Input data.
        service_columns: Columns representing individual add-on
            services, each expected to hold "Yes" when subscribed.

    Returns:
        A new DataFrame with a "ServiceCount" column added. Only
        service columns actually present in the data are counted.
    """
    result = dataframe.copy()
    available_columns = [column for column in service_columns if column in result.columns]
    if not available_columns:
        logger.debug("No service columns found; skipping ServiceCount.")
        return result

    result["ServiceCount"] = (result[available_columns] == "Yes").sum(axis=1)
    return result


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply every engineered feature to a dataset.

    Args:
        dataframe: Input data (already cleaned by
            :mod:`src.preprocessing.cleaner`). Must not include the
            target column's values in any computation — and none of
            the feature functions below reference it.

    Returns:
        A new DataFrame with all applicable engineered feature
        columns added.
    """
    result = dataframe.copy()
    result = add_tenure_group(result)
    result = add_average_monthly_spend(result)
    result = add_estimated_clv(result)
    result = add_contract_risk(result)
    result = add_service_count(result)

    added_columns = [column for column in ENGINEERED_FEATURE_NAMES if column in result.columns]
    logger.info(
        "Feature engineering completed: added %d feature(s): %s", len(added_columns), added_columns
    )
    return result
