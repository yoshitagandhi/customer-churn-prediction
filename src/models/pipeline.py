"""Coordinates the entire model training workflow.

Expected flow::

    Validated Dataset -> Preprocessing -> Sampling -> Model ->
    Hyperparameter Search -> Best Model -> Experiment Tracking ->
    Serialized Artifacts

This module orchestrates that workflow; it contains no model-specific
logic of its own.
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exceptions import ConfigurationError, DataValidationError
from configs.config import settings
from configs.logging_config import get_logger
from src.models.comparison import compare_models, identify_best_model, save_comparison_table
from src.models.experiment_tracker import ExperimentRecord, log_experiment, save_experiment_log
from src.models.registry import get_models
from src.models.serializer import (
    save_all_trained_models,
    save_best_model,
    save_preprocessor_snapshot,
    save_training_metadata,
)
from src.models.trainer import TrainingResult, train_model

logger = get_logger(__name__)


def run_training_pipeline(
    features: pd.DataFrame,
    target: pd.Series,
    model_names: tuple[str, ...] | None = None,
    sampling_strategy: str = settings.default_sampling_strategy,
    validation_size: float = settings.validation_size,
    tune: bool = True,
    positive_label: str = "Yes",
    random_state: int = settings.random_seed,
) -> dict[str, Any]:
    """Run the full training pipeline: split, train every model, compare, serialize.

    Args:
        features: Full feature set (raw, pre-preprocessing).
        target: Full target Series, aligned with ``features``.
        model_names: Registered model names to train. Defaults to
            every registered model (see
            :func:`src.models.registry.get_models`).
        sampling_strategy: Sampling strategy applied to every model in
            this run. Defaults to ``settings.default_sampling_strategy``.
        validation_size: Fraction of data held out for validation.
            Defaults to ``settings.validation_size``.
        tune: Whether to run hyperparameter search for tunable models.
        positive_label: The label representing churn. Defaults to "Yes".
        random_state: Random seed for reproducibility. Defaults to
            ``settings.random_seed``.

    Returns:
        A dictionary with the comparison table, every experiment
        record, every model's TrainingResult, the best model's name,
        and the paths of every artifact saved.
    """
    logger.info("Training started.")
    if features.empty:
        raise DataValidationError(
            "Training features cannot be empty."
    )

    if target.empty:
        raise DataValidationError(
            "Training target cannot be empty."
    )

    if len(features) != len(target):
        raise DataValidationError(
            "Features and target must contain the same number of rows."
    )
    if not model_names:
        raise ConfigurationError(
            "At least one model must be specified."
    )
    
    if not 0 < validation_size < 1:
        raise ConfigurationError(
            "validation_size must be between 0 and 1."
    )
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=validation_size, stratify=target, random_state=random_state
    )
    logger.info(
        "Data split: %d training row(s), %d validation row(s).",
        len(features_train),
        len(features_val),
    )

    experiment_records: list[ExperimentRecord] = []
    training_results: dict[str, TrainingResult] = {}

    for model_name in model_names:
        result = train_model(
            model_name,
            features_train,
            target_train,
            features_val,
            target_val,
            sampling_strategy=sampling_strategy,
            tune=tune,
            positive_label=positive_label,
            random_state=random_state,
        )
        training_results[model_name] = result
        experiment_records.append(
            log_experiment(
                model_name=result.model_name,
                sampling_strategy=result.sampling_strategy,
                hyperparameters=result.best_params,
                validation_metrics=result.validation_metrics,
                training_time_seconds=result.training_time_seconds,
                random_state=random_state,
            )
        )

    comparison_frame = compare_models(experiment_records)
    best_model_info = identify_best_model(comparison_frame)
    best_model_name = best_model_info["model_name"]
    best_training_result = training_results[best_model_name]
    logger.info("Best model identified: '%s'.", best_model_name)

    artifact_paths = {
        "best_model": save_best_model(best_training_result),
        "preprocessor": save_preprocessor_snapshot(best_training_result),
        "training_metadata": save_training_metadata(best_training_result),
        "experiment_log": save_experiment_log(experiment_records),
        "model_comparison": save_comparison_table(comparison_frame),
        "all_models": save_all_trained_models(training_results),
    }
    logger.info("Artifacts saved.")
    logger.info("Training completed.")

    return {
        "comparison_table": comparison_frame,
        "experiment_records": experiment_records,
        "training_results": training_results,
        "best_model_name": best_model_name,
        "artifact_paths": artifact_paths,
        # Exposed so downstream milestones (evaluation, explainability,
        # threshold optimization) can reuse the EXACT same split rather
        # than re-splitting the data themselves.
        "features_train": features_train,
        "target_train": target_train,
        "features_val": features_val,
        "target_val": target_val,
    }
