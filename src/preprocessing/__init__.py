"""Preprocessing and feature engineering.

Expose the public preprocessing API and provide a convenience
function for running the complete preprocessing workflow.

This package converts a validated dataset into a machine-learning-ready
feature matrix through cleaning, feature engineering, imputation,
encoding, and scaling. The resulting preprocessing pipeline is fully
serializable and reusable for both training and inference.

Warning:
    ``run_preprocessing_pipeline`` fits the preprocessing pipeline on
    the entire dataset for demonstration and artifact generation.
    During model training, always fit the preprocessor using only the
    training split to avoid data leakage.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.preprocessing.cleaner import (
    apply_row_level_cleaning,
    clean_dataset,
)
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.metadata import (
    build_feature_metadata,
    save_feature_metadata,
)
from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_dataset,
)
from src.preprocessing.transformer import build_transformer
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)

__all__ = [
    "clean_dataset",
    "apply_row_level_cleaning",
    "engineer_features",
    "build_transformer",
    "build_preprocessing_pipeline",
    "fit_preprocessor",
    "transform_dataset",
    "save_preprocessor",
    "load_preprocessor",
    "build_feature_metadata",
    "save_feature_metadata",
    "run_preprocessing_pipeline",
]


def run_preprocessing_pipeline(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
) -> dict[str, Any]:
    """Run the complete preprocessing workflow."""

    if dataframe.empty:
        raise DataValidationError(
            "Cannot preprocess an empty dataset."
        )

    logger.info("Starting preprocessing pipeline.")

    cleaned_dataset = clean_dataset(dataframe)

    if target_column not in cleaned_dataset.columns:
        logger.warning(
            "Target column '%s' not found in dataset.",
            target_column,
        )

    feature_frame = cleaned_dataset.drop(
        columns=[target_column],
        errors="ignore",
    )

    logger.info("Building preprocessing pipeline.")
    pipeline = build_preprocessing_pipeline()

    logger.info("Fitting preprocessing pipeline.")
    pipeline = fit_preprocessor(
        pipeline,
        feature_frame,
    )

    logger.info("Transforming dataset.")
    processed_features = transform_dataset(
        pipeline,
        feature_frame,
    )

    logger.info("Saving preprocessing pipeline.")
    preprocessor_path = save_preprocessor(
        pipeline,
    )

    logger.info("Generating feature metadata.")
    metadata = build_feature_metadata(
        pipeline,
        input_feature_names=list(
            feature_frame.columns,
        ),
    )

    metadata_paths = save_feature_metadata(
        metadata,
    )

    logger.info(
        "Preprocessing completed successfully "
        "(%d input features → %d processed features).",
        feature_frame.shape[1],
        processed_features.shape[1],
    )

    return {
        "pipeline": pipeline,
        "cleaned_dataset": cleaned_dataset,
        "processed_features": processed_features,
        "processed_shape": processed_features.shape,
        "metadata": metadata,
        "preprocessor_path": preprocessor_path,
        "metadata_paths": metadata_paths,
    }

