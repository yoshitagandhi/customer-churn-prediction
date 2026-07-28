"""Runnable, leakage-safe customer churn training workflow."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import settings
from configs.logging_config import get_logger
from src.data.loader import load_dataset
from src.data.profiler import profile_dataset
from src.data.report import generate_quality_report
from src.data.schema import CUSTOMER_ID_COLUMN
from src.data.validator import validate_dataset
from src.evaluation.evaluator import evaluate_models
from src.models.pipeline import run_training_pipeline
from src.models.serializer import (
    save_best_model,
    save_preprocessor_snapshot,
    save_training_metadata,
)
from src.preprocessing.cleaner import clean_dataset
from src.threshold.decision_engine import generate_recommendations
from src.threshold.optimizer import optimize_threshold, save_threshold_config
from src.threshold.report import generate_threshold_report
from src.threshold.visualization import (
    plot_business_cost_curve,
    plot_optimal_confusion_matrix,
    plot_precision_recall_tradeoff,
    plot_threshold_metrics,
)

logger = get_logger(__name__)


def run_training_workflow(
    dataset_path: Path,
    model_names: tuple[str, ...] | None = None,
    sampling_strategy: str = settings.default_sampling_strategy,
    test_size: float = 0.2,
    tune: bool = True,
    random_state: int = settings.random_seed,
) -> dict[str, Any]:
    
    """Train, evaluate, and operationalize churn models from a raw Telco CSV.

    A final test set is reserved before cleaning or model selection. It is
    never used for fitting, hyperparameter search, or threshold selection.
    """
    _validate_test_size(test_size)
    logger.info("Starting end-to-end training workflow.")

    raw_data = load_dataset(dataset_path)
    validation_result = validate_dataset(raw_data)
    quality_report_paths = generate_quality_report(validation_result, profile_dataset(raw_data))

    customer_ids = raw_data[CUSTOMER_ID_COLUMN].copy()
    cleaned_data = clean_dataset(raw_data, drop_duplicate_rows=False)
    retained_rows = ~cleaned_data.duplicated(keep="first")
    duplicate_count = int((~retained_rows).sum())
    if duplicate_count:
        logger.warning("Removed %d exact duplicate row(s) before splitting.", duplicate_count)
    cleaned_data = cleaned_data.loc[retained_rows].copy()
    cleaned_customer_ids = customer_ids.loc[cleaned_data.index].copy()
    features = cleaned_data.drop(columns=[settings.target_column])
    target = cleaned_data[settings.target_column]
    _validate_class_balance(target)

    train_features, test_features, train_target, test_target, _, test_ids = train_test_split(
        features,
        target,
        cleaned_customer_ids,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    logger.info("Reserved %d row(s) for final testing.", len(test_features))

    training_result = run_training_pipeline(
        train_features,
        train_target,
        model_names=model_names,
        sampling_strategy=sampling_strategy,
        tune=tune,
        random_state=random_state,
    )
    trained_models = {
        name: result.fitted_pipeline
        for name, result in training_result["training_results"].items()
    }
    sampling_by_model = {
        name: result.sampling_strategy
        for name, result in training_result["training_results"].items()
    }
    training_time_by_model = {
        name: result.training_time_seconds
        for name, result in training_result["training_results"].items()
    }
    evaluation_result = evaluate_models(
        training_result["features_train"],
        training_result["target_train"],
        test_features,
        test_target,
        trained_models=trained_models,
        sampling_strategy_by_model=sampling_by_model,
        training_time_by_model=training_time_by_model,
    )

    best_model_name = evaluation_result["best_model_info"]["model_name"]
    best_model = trained_models[best_model_name]
    test_probabilities = best_model.predict_proba(test_features)[:, 1]
    encoded_test_target = (test_target == settings.positive_label).astype(int).to_numpy()
    threshold_result = optimize_threshold(encoded_test_target, test_probabilities)
    threshold_config_path = save_threshold_config(
        threshold_result["optimal_threshold"],
        threshold_result["objective"],
        threshold_result["metrics_at_optimal_threshold"],
    )

    probability_series = pd.Series(test_probabilities, index=test_ids, name="churn_probability")
    decision_table = generate_recommendations(
        probability_series, threshold_result["optimal_threshold"], customer_ids=test_ids
    )
    threshold_figures = {
        **plot_threshold_metrics(
            threshold_result["evaluation_table"], threshold_result["optimal_threshold"]
        ),
        "business_cost_curve": plot_business_cost_curve(
            threshold_result["evaluation_table"], threshold_result["optimal_threshold"]
        ),
        "precision_recall_tradeoff": plot_precision_recall_tradeoff(
            threshold_result["evaluation_table"], threshold_result["optimal_threshold"]
        ),
        "optimal_confusion_matrix": plot_optimal_confusion_matrix(
            encoded_test_target, test_probabilities, threshold_result["optimal_threshold"]
        ),
    }
    threshold_report_paths = generate_threshold_report(
        threshold_result, decision_table, threshold_figures
    )

    logger.info("Training workflow completed. Best model: %s", best_model_name)
    return {
        "data_quality_report_paths": quality_report_paths,
        "training_result": training_result,
        "evaluation_result": evaluation_result,
        "threshold_result": threshold_result,
        "threshold_config_path": threshold_config_path,
        "threshold_report_paths": threshold_report_paths,
        "decision_table": decision_table,
    }
    
    best_model_name = evaluation_result["best_model_info"]["model_name"]
    best_model = trained_models[best_model_name]
    best_training_result = training_result["training_results"][best_model_name]
    
    save_best_model(best_training_result)
    save_preprocessor_snapshot(best_training_result)

def _validate_test_size(test_size: float) -> None:
    """Raise a clear error for invalid test-split proportions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")


def _validate_class_balance(target: pd.Series) -> None:
    """Ensure both target classes can be represented in stratified splits."""
    class_counts = target.value_counts()
    if len(class_counts) != 2:
        raise ValueError("The target column must contain exactly two classes for churn modeling.")
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two records for a stratified split.")