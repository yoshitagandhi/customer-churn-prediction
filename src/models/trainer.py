"""Model-agnostic training.

Combines the Milestone 4/5 preprocessing + sampling pipeline with a
registered model into a single pipeline, optionally tunes it via
:mod:`src.models.hyperparameter`, fits it on the training fold, and
evaluates it on the validation fold. Contains no model-specific
branching beyond reading each estimator's own supported parameters.
"""

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from configs.config import settings
from configs.logging_config import get_logger
from src.models.hyperparameter import tune_model
from src.models.registry import ModelSpec, build_estimator, get_model_spec
from src.sampling import (
    build_sampling_pipeline,
    compute_scale_pos_weight,
    get_class_weight_parameter,
)

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Outcome of training a single model.

    Attributes:
        model_name: Registered model name.
        sampling_strategy: Sampling strategy used for this run.
        fitted_pipeline: The complete fitted pipeline (preprocessing +
            sampling + model). Sampling is automatically skipped
            during ``.predict()``/``.predict_proba()``, so this
            pipeline alone is sufficient for inference on raw
            feature data.
        best_params: The model step's parameter configuration used.
        validation_metrics: Metrics computed on the held-out
            validation fold.
        training_time_seconds: Total wall-clock time for this run.
    """

    model_name: str
    sampling_strategy: str
    fitted_pipeline: Any
    best_params: dict[str, Any]
    validation_metrics: dict[str, float]
    training_time_seconds: float


def compute_validation_metrics(
    target_true: Any, target_pred: Any, target_proba: Any
) -> dict[str, float]:
    """Compute the standard set of validation metrics for a binary classifier.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_pred: Predicted labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.

    Returns:
        A dictionary with roc_auc, precision, recall, f1, and pr_auc.
    """
    return {
        "accuracy": round(float(accuracy_score(target_true, target_pred)), 4),
        "roc_auc": round(float(roc_auc_score(target_true, target_proba)), 4),
        "precision": round(float(precision_score(target_true, target_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(target_true, target_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(target_true, target_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(target_true, target_proba)), 4),
    }


def _encode_target(target: pd.Series, positive_label: str) -> pd.Series:
    """Encode a string target Series to 0/1, with ``positive_label`` mapped to 1.

    Args:
        target: Raw target Series (e.g., "Yes"/"No").
        positive_label: The label to encode as 1.

    Returns:
        A Series of 0/1 integers.
    """
    return (target == positive_label).astype(int)


def _apply_algorithm_level_balancing(
    estimator: Any,
    model_spec: ModelSpec,
    sampling_strategy: str,
    target_train: pd.Series,
    positive_label: str,
) -> Any:
    """Configure an estimator for algorithm-level class balancing, if applicable.

    Reuses :mod:`src.sampling.samplers` utilities rather than
    recomputing class-weighting logic. Strategies that a given
    estimator does not support are skipped with a warning instead of
    raising, since not every model exposes every balancing parameter.

    Args:
        estimator: An unfitted estimator instance.
        model_spec: The model's registry specification (used for
            logging only).
        sampling_strategy: The requested sampling strategy.
        target_train: Raw (string-labeled) training target, used to
            compute scale_pos_weight if requested.
        positive_label: The label representing the positive
            (churned) class.

    Returns:
        The same estimator, with balancing parameters set if
        applicable.
    """
    supported_params = estimator.get_params()

    if sampling_strategy == "class_weight":
        if "class_weight" in supported_params:
            estimator.set_params(class_weight=get_class_weight_parameter())
        else:
            logger.warning(
                "Model '%s' does not support class_weight; ignoring.", model_spec.name
            )
    elif sampling_strategy == "scale_pos_weight":
        if "scale_pos_weight" in supported_params:
            ratio = compute_scale_pos_weight(target_train, positive_label)
            estimator.set_params(scale_pos_weight=ratio)
        else:
            logger.warning(
                "Model '%s' does not support scale_pos_weight; ignoring.", model_spec.name
            )

    return estimator


def _build_full_pipeline(
    sampling_strategy: str,
    estimator: Any,
    random_state: int,
) -> Any:
    """Append a model as the final step of the preprocessing + sampling pipeline.

    Args:
        sampling_strategy: Sampling strategy identifier.
        estimator: The (possibly balancing-configured) estimator to
            append as the pipeline's final step.
        random_state: Random seed for reproducible resampling.

    Returns:
        A single ``imblearn.pipeline.Pipeline`` combining
        preprocessing, sampling (if any), and the model.
    """

    sampling_pipeline = build_sampling_pipeline(
        strategy=sampling_strategy,
        random_state=random_state,
    )

    steps = [
        *sampling_pipeline.steps,
        ("model", estimator),
    ]

    return ImbPipeline(steps=steps)
    

def train_model(
    model_name: str,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_val: pd.DataFrame,
    target_val: pd.Series,
    sampling_strategy: str = settings.default_sampling_strategy,
    tune: bool = True,
    positive_label: str = "Yes",
    random_state: int = settings.random_seed,
) -> TrainingResult:
    """Train a single registered model end to end and evaluate it on validation data.

    Builds the full preprocessing + sampling + model pipeline,
    optionally tunes it via ``RandomizedSearchCV`` (which resamples
    fresh inside every CV fold), fits it on the training fold, and
    evaluates it on the untouched validation fold.

    Args:
        model_name: Registered model name (see
            :func:`src.models.registry.get_models`).
        features_train: Training features (raw, pre-preprocessing).
        target_train: Training labels (raw, e.g. "Yes"/"No").
        features_val: Validation features (raw, pre-preprocessing).
            Never resampled.
        target_val: Validation labels (raw).
        sampling_strategy: Sampling strategy to use. Defaults to
            ``settings.default_sampling_strategy``.
        tune: Whether to run hyperparameter search for tunable
            models. Non-tunable models always train once with their
            registry default parameters.
        positive_label: The label representing churn. Defaults to
            "Yes".
        random_state: Random seed for reproducibility. Defaults to
            ``settings.random_seed``.

    Returns:
        A TrainingResult with the fitted pipeline and validation
        metrics.
    """
    logger.info("Training started: model='%s', sampling='%s'.", model_name, sampling_strategy)
    start_time = time.perf_counter()

    target_train_encoded = _encode_target(target_train, positive_label)
    target_val_encoded = _encode_target(target_val, positive_label)

    model_spec = get_model_spec(model_name)
    estimator = build_estimator(model_name)
    estimator = _apply_algorithm_level_balancing(
        estimator, model_spec, sampling_strategy, target_train, positive_label
    )

    full_pipeline = _build_full_pipeline(sampling_strategy, estimator, random_state)
    logger.info("Pipeline initialized.")

    if tune and model_spec.tunable:
        logger.info("Hyperparameter search started.")
        search_result = tune_model(
            model_name,
            full_pipeline,
            features_train,
            target_train_encoded,
            random_state=random_state,
        )
        fitted_pipeline = search_result.best_pipeline
        best_params = search_result.best_params
        logger.info("Best parameters found: %s", best_params)
    else:
        fitted_pipeline = full_pipeline.fit(features_train, target_train_encoded)
        best_params = fitted_pipeline.named_steps["model"].get_params()

    logger.info("Model trained.")

    target_pred = fitted_pipeline.predict(features_val)
    target_proba = fitted_pipeline.predict_proba(features_val)[:, 1]
    validation_metrics = compute_validation_metrics(target_val_encoded, target_pred, target_proba)

    training_time = round(time.perf_counter() - start_time, 4)
    logger.info("Training completed: model='%s' (%.2fs).", model_name, training_time)

    return TrainingResult(
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        fitted_pipeline=fitted_pipeline,
        best_params=best_params,
        validation_metrics=validation_metrics,
        training_time_seconds=training_time,
    )
