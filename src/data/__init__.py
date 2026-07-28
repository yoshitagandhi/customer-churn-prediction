"""Data ingestion, validation, and data quality reporting.

This package loads the raw customer churn dataset, validates it
against the expected schema, profiles its structure, and generates
data quality reports. It performs no cleaning, imputation, encoding,
or feature engineering -- see `src.preprocessing` and `src.features`
for those concerns in later milestones.

Typical usage:

    from src.data import run_data_quality_pipeline

    dataframe, validation_result, profile = run_data_quality_pipeline()
"""

from pathlib import Path

import pandas as pd

from configs.logging_config import get_logger
from src.data.loader import load_dataset
from src.data.profiler import profile_dataset
from src.data.report import generate_quality_report
from src.data.validator import ValidationResult, validate_dataset

logger = get_logger(__name__)

__all__ = [
    "load_dataset",
    "validate_dataset",
    "profile_dataset",
    "generate_quality_report",
    "run_data_quality_pipeline",
    "ValidationResult",
]


def run_data_quality_pipeline(
    file_path: Path | None = None,
) -> tuple[pd.DataFrame, ValidationResult, dict]:
    """Run the full ingestion pipeline: load, validate, profile, report.

    This is a thin orchestration convenience function that wires
    together `loader`, `validator`, `profiler`, and `report` exactly
    as described in the milestone's pipeline flow. Each stage remains
    independently usable and testable.

    Args:
        file_path: Path to the dataset file. Defaults to the
            configured default dataset path when not provided.

    Returns:
        A tuple of (validated DataFrame, ValidationResult, profile
        dictionary). The DataFrame is returned unmodified.
    """
    dataframe = load_dataset(file_path)
    validation_result = validate_dataset(dataframe)
    profile = profile_dataset(dataframe)
    generate_quality_report(validation_result, profile)

    logger.info("Data quality pipeline completed.")
    return dataframe, validation_result, profile
