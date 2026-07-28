"""Model training, hyperparameter optimization, and experiment tracking.

Trains every registered model through the same leakage-safe
preprocessing + sampling pipeline (Milestones 4 and 5), tunes tunable
models via RandomizedSearchCV, tracks every run, compares results
consistently, and serializes the best-performing model.

This package performs no explainability (SHAP), threshold
optimization, or deployment logic -- those are later milestones.

Typical usage::

    from src.data import run_data_quality_pipeline
    from src.preprocessing import clean_dataset
    from src.models import run_training_pipeline

    dataframe, _, _ = run_data_quality_pipeline()
    cleaned = clean_dataset(dataframe)
    features = cleaned.drop(columns=["Churn"])
    target = cleaned["Churn"]

    results = run_training_pipeline(features, target)
"""

from src.models.comparison import compare_models, identify_best_model, save_comparison_table
from src.models.experiment_tracker import (
    ExperimentRecord,
    load_experiment_log,
    log_experiment,
    save_experiment_log,
)
from src.models.hyperparameter import (
    HyperparameterSearchResult,
    get_search_space,
    list_tunable_models,
    tune_model,
)
from src.models.pipeline import run_training_pipeline
from src.models.registry import ModelSpec, build_estimator, get_model_spec, get_models
from src.models.serializer import (
    load_all_trained_models,
    load_model,
    load_training_metadata,
    save_all_trained_models,
    save_best_model,
    save_preprocessor_snapshot,
    save_training_metadata,
)
from src.models.trainer import TrainingResult, compute_validation_metrics, train_model

__all__ = [
    "get_models",
    "get_model_spec",
    "build_estimator",
    "ModelSpec",
    "train_model",
    "compute_validation_metrics",
    "TrainingResult",
    "tune_model",
    "get_search_space",
    "list_tunable_models",
    "HyperparameterSearchResult",
    "log_experiment",
    "save_experiment_log",
    "load_experiment_log",
    "ExperimentRecord",
    "compare_models",
    "identify_best_model",
    "save_comparison_table",
    "save_best_model",
    "save_preprocessor_snapshot",
    "save_training_metadata",
    "save_all_trained_models",
    "load_model",
    "load_training_metadata",
    "load_all_trained_models",
    "run_training_pipeline",
]
