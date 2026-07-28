"""Main evaluation pipeline.

Coordinates the full evaluation workflow across every trained model.
Contains no metric-calculation or plotting logic itself — those live
in :mod:`src.evaluation.metrics` and
:mod:`src.evaluation.visualizations` respectively. This module never
retrains a model, reruns hyperparameter search, or refits
preprocessing; it only calls ``.predict()``/``.predict_proba()`` on
already-fitted pipelines.
"""

from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix

from configs.config import settings
from configs.logging_config import get_logger
from src.evaluation.calibration import generate_calibration_curve
from src.evaluation.comparison import compare_models, identify_best_model
from src.evaluation.learning_curve import generate_learning_curve
from src.evaluation.metrics import compute_metrics
from src.evaluation.report import generate_evaluation_report
from src.evaluation.visualizations import (
    plot_classification_report_heatmap,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.models import load_all_trained_models

logger = get_logger(__name__)


def _encode_target(target: pd.Series, positive_label: str) -> pd.Series:
    """Encode a string target Series to 0/1, with ``positive_label`` mapped to 1."""
    return (target == positive_label).astype(int)

def evaluate_model(
    model,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a single fitted production model.
    Used by the Streamlit dashboard.
    """

    target_encoded = _encode_target(target, settings.positive_label)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics = compute_metrics(
        target_true=target_encoded.to_numpy(),
        target_pred=predictions,
        target_proba=probabilities,
    )

    calibration = generate_calibration_curve(
        target_true=target_encoded.to_numpy(),
        target_proba=probabilities,
        model_name="Production Model",
    )

    return {
        "metrics": metrics,
        "calibration": calibration,
    }
def evaluate_models(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    trained_models: dict[str, Any] | None = None,
    model_names: tuple[str, ...] | None = None,
    sampling_strategy_by_model: dict[str, str] | None = None,
    training_time_by_model: dict[str, float] | None = None,
    positive_label: str = settings.positive_label,
) -> dict[str, Any]:
    """Evaluate every trained model consistently on the same test data.

    Args:
        features_train: Training features used to train the models in
            Milestone 6. Never refit here — used only to generate the
            best model's learning curve.
        target_train: Training labels aligned with ``features_train``.
        features_test: Held-out test features. Every model is
            evaluated on this same set, ensuring a fair comparison.
        target_test: Held-out test labels aligned with ``features_test``.
        trained_models: Mapping of model name to its already-fitted
            pipeline (e.g., from Milestone 6's
            ``run_training_pipeline`` result). If not provided, models
            are loaded from disk via
            :func:`src.models.load_all_trained_models`.
        model_names: Which registered models to evaluate, used only
            when loading from disk (``trained_models`` is None).
        sampling_strategy_by_model: Mapping of model name to its
            sampling strategy, for the comparison table.
        training_time_by_model: Mapping of model name to its training
            duration (seconds), sourced from Milestone 6's experiment
            log (this milestone does not time any training itself).
        positive_label: The label representing churn. Defaults to
            ``settings.positive_label``.

    Returns:
        A dictionary with per-model metrics, the ranked comparison
        table, the best model's info (with selection reason), every
        generated figure path, the best model's calibration result,
        and the generated report paths.

    Raises:
        ValueError: If ``trained_models`` is not provided and
            ``model_names`` is also missing.
    """
    logger.info("Evaluation started.")

    if trained_models is None:
        if model_names is None:
            raise ValueError("model_names is required when trained_models is not provided.")
        trained_models = load_all_trained_models(model_names)
    logger.info("Models loaded.")

    sampling_strategy_by_model = sampling_strategy_by_model or {}
    training_time_by_model = training_time_by_model or {}
    target_test_encoded = _encode_target(target_test, positive_label)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    proba_by_model: dict[str, Any] = {}
    pred_by_model: dict[str, Any] = {}

    for model_name, fitted_pipeline in trained_models.items():
        target_pred = fitted_pipeline.predict(features_test)
        target_proba = fitted_pipeline.predict_proba(features_test)[:, 1]
        metrics_by_model[model_name] = compute_metrics(
            target_test_encoded, target_pred, target_proba
        )
        pred_by_model[model_name] = target_pred
        proba_by_model[model_name] = target_proba
    logger.info("Metrics calculated.")

    comparison_frame = compare_models(
        metrics_by_model, sampling_strategy_by_model, training_time_by_model
    )
    best_model_info = identify_best_model(comparison_frame)
    best_model_name = best_model_info["model_name"]
    best_pipeline = trained_models[best_model_name]

    curve_inputs = {
        model_name: (target_test_encoded, proba_by_model[model_name])
        for model_name in trained_models
    }
    figure_paths: dict[str, Any] = {
        "roc_curve": plot_roc_curve(curve_inputs),
        "precision_recall_curve": plot_precision_recall_curve(curve_inputs),
        "metric_comparison": plot_metric_comparison(comparison_frame),
        "classification_report_heatmap": plot_classification_report_heatmap(
            {best_model_name: metrics_by_model[best_model_name]}
        ),
    }
    for model_name in trained_models:
        figure_paths[f"confusion_matrix_{model_name}"] = plot_confusion_matrix(
            target_test_encoded, pred_by_model[model_name], model_name
        )
    logger.info("Visualizations generated.")

    calibration_result = generate_calibration_curve(
        target_test_encoded, proba_by_model[best_model_name], best_model_name
    )
    figure_paths["calibration_curve"] = calibration_result["figure_path"]
    figure_paths["learning_curve"] = generate_learning_curve(
        best_pipeline, features_train, target_train, best_model_name
    )

    report_paths = generate_evaluation_report(
        comparison_frame, best_model_info, metrics_by_model, figure_paths, calibration_result
    )
    logger.info("Reports created.")
    logger.info("Evaluation completed.")

    return {
        "metrics_by_model": metrics_by_model,
        "comparison_table": comparison_frame,
        "best_model_info": best_model_info,
        "figure_paths": figure_paths,
        "calibration_result": calibration_result,
        "report_paths": report_paths,
    }

def evaluate_models(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    positive_label: str = settings.positive_label,
) -> dict[str, Any]:
    """
    Evaluate one fitted model.

    Used by the Streamlit dashboard.
    """

    target_encoded = _encode_target(target, positive_label)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics = compute_metrics(
        target_encoded,
        predictions,
        probabilities,
    )

    confusion = confusion_matrix(
        target_encoded,
        predictions,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "predictions": predictions,
        "probabilities": probabilities,
    }
