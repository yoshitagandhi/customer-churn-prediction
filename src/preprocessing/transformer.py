"""Reusable preprocessing transformer components.

This module only *defines* scikit-learn transformer objects — it
never calls ``.fit()`` on anything. Fitting happens in
:mod:`src.preprocessing.pipeline`, where the caller controls exactly
what data the transformer is fit on (preventing data leakage).

Features are grouped into three treatments, as required by this
milestone:

- **Numerical** (continuous): median imputation + ``StandardScaler``.
- **Binary** (already 0/1): most-frequent imputation only, no
  scaling, so the values stay interpretable as 0/1.
- **Categorical**: most-frequent imputation + ``OneHotEncoder`` with
  ``handle_unknown="ignore"`` so inference never breaks on an unseen
  category.
"""

from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import settings
from configs.logging_config import get_logger
from src.data.schema import CATEGORICAL_COLUMNS, CUSTOMER_ID_COLUMN

logger = get_logger(__name__)

# Raw categorical columns from the dataset schema, excluding the
# identifier (dropped during cleaning) and the target (never a
# feature).
_BASE_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = tuple(
    column
    for column in CATEGORICAL_COLUMNS
    if column not in (CUSTOMER_ID_COLUMN, settings.target_column)
)

# Categorical features engineered in src.preprocessing.feature_engineering.
_ENGINEERED_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("TenureGroup", "ContractRisk")

# Full feature contract this preprocessing pipeline expects, once
# cleaning and feature engineering have run. This is the single
# source of truth for which column goes through which treatment.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    _BASE_CATEGORICAL_FEATURES + _ENGINEERED_CATEGORICAL_FEATURES
)
BINARY_FEATURES: Final[tuple[str, ...]] = ("SeniorCitizen",)
NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend",
    "EstimatedCLV",
    "ServiceCount",
)


def build_numerical_transformer() -> Pipeline:
    """Build the transformer for continuous numerical features.

    Returns:
        An unfitted Pipeline: median imputation followed by
        standardization.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_binary_transformer() -> Pipeline:
    """Build the transformer for binary (already 0/1) features.

    Only imputes missing values; scaling is intentionally skipped so
    binary features remain interpretable as 0/1.

    Returns:
        An unfitted Pipeline containing only most-frequent imputation.
    """
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])


def build_categorical_transformer() -> Pipeline:
    """Build the transformer for categorical features.

    Returns:
        An unfitted Pipeline: most-frequent imputation followed by
        one-hot encoding that safely ignores unseen categories at
        inference time.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_transformer(
    numerical_features: tuple[str, ...] = NUMERICAL_FEATURES,
    binary_features: tuple[str, ...] = BINARY_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build the unfitted ColumnTransformer combining every feature group.

    This function only defines the transformer; it does not fit it.

    Args:
        numerical_features: Continuous numeric columns. Defaults to
            ``NUMERICAL_FEATURES``.
        binary_features: Already-binary columns. Defaults to
            ``BINARY_FEATURES``.
        categorical_features: Categorical columns. Defaults to
            ``CATEGORICAL_FEATURES``.

    Returns:
        An unfitted ColumnTransformer with clean (non-prefixed) output
        feature names.
    """
    logger.debug(
        "Building ColumnTransformer: %d numerical, %d binary, %d categorical feature(s).",
        len(numerical_features),
        len(binary_features),
        len(categorical_features),
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", build_numerical_transformer(), list(numerical_features)),
            ("binary", build_binary_transformer(), list(binary_features)),
            ("categorical", build_categorical_transformer(), list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
