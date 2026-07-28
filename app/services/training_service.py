"""
===============================================================================
Customer Churn Prediction Platform
Training Service

File        : training_service.py
Version     : 1.0

Purpose
-------
Provides the business-facing training API for the Streamlit application.

Responsibilities
----------------
• Execute the end-to-end training workflow
• Load datasets
• Train and evaluate models
• Persist trained artifacts
• Load training metadata
• Expose available trained models

Notes
-----
• Never performs preprocessing itself.
• Never builds ML pipelines.
• Never tunes hyperparameters.
• Never contains Streamlit code.
• Delegates all work to the backend packages.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.logging_config import get_logger

from src.data.loader import load_dataset
from src.models.pipeline import train_model_pipeline
from src.models.serializer import (
    load_all_models,
    load_training_metadata,
)
from src.utils.exceptions import (
    ConfigurationError,
    DataValidationError,
)

logger = get_logger(__name__)

@dataclass(slots=True)
class TrainingServiceResult:
    """
    Result returned after a complete training run.
    """

    best_model_name: str
    best_model: Any
    metrics: dict[str, Any]
    experiment_log: list[Any]
    training_time: float
    metadata: dict[str, Any]

def _validate_dataset_path(
    dataset_path: str | Path,
) -> Path:
    """
    Validate dataset path.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise DataValidationError(
            f"Dataset not found: {path}"
        )

    return path.resolve()

def _validate_target_column(
    dataset,
    target_column: str,
) -> None:
    """
    Validate target column.
    """

    if target_column not in dataset.columns:
        raise DataValidationError(
            f"Target column '{target_column}' does not exist."
        )

def train_pipeline(
    *,
    dataset_path: str | Path,
    target_column: str,
) -> TrainingServiceResult:
    """
    Execute the complete training workflow.
    """

    logger.info(
        "Starting training pipeline."
    )

    dataset_path = _validate_dataset_path(
        dataset_path
    )

    dataset = load_dataset(dataset_path)

    _validate_target_column(
        dataset,
        target_column,
    )

    training_result = train_model_pipeline(
        dataframe=dataset,
        target_column=target_column,
    )

    metadata = load_training_metadata()

    logger.info(
        "Training pipeline completed."
    )

    return TrainingServiceResult(
        best_model_name=training_result.best_model_name,
        best_model=training_result.best_model,
        metrics=training_result.metrics,
        experiment_log=training_result.experiment_records,
        training_time=training_result.training_time,
        metadata=metadata,
    )

def retrain_pipeline(
    *,
    dataset_path: str | Path,
    target_column: str,
) -> TrainingServiceResult:
    """
    Retrain all models using a new dataset.

    This is currently identical to train_pipeline()
    but is exposed separately for UI clarity and
    future extensibility.
    """

    logger.info(
        "Retraining pipeline."
    )

    return train_pipeline(
        dataset_path=dataset_path,
        target_column=target_column,
    )

def get_available_models() -> list[str]:
    """
    Return names of all serialized models.
    """

    logger.info(
        "Loading trained models."
    )

    models = load_all_models()

    return sorted(models.keys())

def get_training_metadata() -> dict[str, Any]:
    """
    Load metadata for the latest training run.
    """

    logger.info(
        "Loading training metadata."
    )

    metadata = load_training_metadata()

    if not metadata:
        raise ConfigurationError(
            "Training metadata could not be loaded."
        )

    return metadata

def get_training_status() -> dict[str, Any]:
    """
    Return a lightweight summary of the
    latest training session.
    """

    metadata = get_training_metadata()

    return {
        "trained_model": metadata.get(
            "best_model"
        ),
        "training_time": metadata.get(
            "training_time"
        ),
        "training_timestamp": metadata.get(
            "timestamp"
        ),
        "project_version": metadata.get(
            "project_version"
        ),
    }

__all__ = [
    "TrainingServiceResult",
    "train_pipeline",
    "retrain_pipeline",
    "get_available_models",
    "get_training_metadata",
    "get_training_status",
]