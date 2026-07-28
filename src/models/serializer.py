"""Model and metadata serialization.

Saves everything needed for inference: the complete fitted pipeline
(preprocessing + model, with sampling automatically skipped at
predict time), a standalone snapshot of just the fitted preprocessing
step, and structured training metadata.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from configs.config import settings
from configs.logging_config import get_logger
from src.models.trainer import TrainingResult

logger = get_logger(__name__)


def save_best_model(training_result: TrainingResult, path: Path = settings.best_model_path) -> Path:
    """Serialize the complete fitted pipeline for the best-performing model.

    The saved object already includes fitted preprocessing, so it can
    be loaded and used to call ``.predict()``/``.predict_proba()``
    directly on raw feature data — no other artifact is required for
    inference. Any sampling step is automatically skipped during
    prediction.

    Args:
        training_result: The winning model's TrainingResult.
        path: Destination path. Defaults to ``settings.best_model_path``.

    Returns:
        The path the model was saved to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_result.fitted_pipeline, path)
    logger.info("Best model serialized to %s.", path)
    return path

from sklearn.pipeline import Pipeline

def save_preprocessor_snapshot(
    training_result: TrainingResult,
    path: Path = settings.preprocessor_path,
) -> Path:
    """
    Serialize the fitted preprocessing pipeline.
    """

    fitted_pipeline = training_result.fitted_pipeline

    preprocessing_steps = []

    for name, step in fitted_pipeline.steps:
        if name in ("sampling", "model"):
            break
        preprocessing_steps.append((name, step))

    preprocessing_pipeline = Pipeline(preprocessing_steps)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessing_pipeline, path)

    logger.info("Preprocessing pipeline serialized to %s.", path)
    return path


def save_training_metadata(
    training_result: TrainingResult, path: Path = settings.training_metadata_path
) -> Path:
    """Save structured metadata describing the winning training run.

    Args:
        training_result: The winning model's TrainingResult.
        path: Destination path. Defaults to
            ``settings.training_metadata_path``.

    Returns:
        The path the metadata was saved to.
    """
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": training_result.model_name,
        "sampling_strategy": training_result.sampling_strategy,
        "best_params": training_result.best_params,
        "validation_metrics": training_result.validation_metrics,
        "training_time_seconds": training_result.training_time_seconds,
        "random_seed": settings.random_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Training metadata saved to %s.", path)
    return path


def save_all_trained_models(
    training_results: dict[str, TrainingResult], directory: Path = settings.experiments_dir
) -> dict[str, Path]:
    """Serialize every trained model's fitted pipeline, not just the best one.

    Milestone 6 only needs the single best model for production use
    (see :func:`save_best_model`), but Milestone 7's multi-model
    evaluation (per-model confusion matrices, ROC/PR overlays, model
    comparison) needs every trained model's fitted pipeline available
    on disk too. This is purely additive: it does not change what
    :func:`save_best_model` or any other function saves.

    Args:
        training_results: Mapping of model name to its TrainingResult,
            as returned in ``run_training_pipeline``'s result dict.
        directory: Destination directory. Defaults to
            ``settings.experiments_dir``.

    Returns:
        A mapping of model name to the path its fitted pipeline was
        saved to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for model_name, result in training_results.items():
        model_path = directory / f"{model_name}.pkl"
        joblib.dump(result.fitted_pipeline, model_path)
        paths[model_name] = model_path
    logger.info("Saved %d trained model(s) to %s.", len(paths), directory)
    return paths


def load_all_trained_models(
    model_names: tuple[str, ...], directory: Path = settings.experiments_dir
) -> dict[str, Any]:
    """Load every trained model's fitted pipeline previously saved by name.

    Args:
        model_names: Registered model names to load.
        directory: Source directory. Defaults to
            ``settings.experiments_dir``.

    Returns:
        A mapping of model name to its deserialized, fitted pipeline.

    Raises:
        FileNotFoundError: If any requested model's file is missing.
    """
    models: dict[str, Any] = {}
    for model_name in model_names:
        model_path = directory / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        models[model_name] = joblib.load(model_path)
    logger.info("Loaded %d trained model(s) from %s.", len(models), directory)
    return models


def load_model(path: Path = settings.best_model_path) -> Any:
    """Load a previously serialized model pipeline.

    Args:
        path: Location of the serialized model. Defaults to
            ``settings.best_model_path``.

    Returns:
        The deserialized, fitted pipeline, ready for
        ``.predict()``/``.predict_proba()`` on raw feature data.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s.", path)
    return model


def load_training_metadata(path: Path = settings.training_metadata_path) -> dict[str, Any]:
    """Load previously saved training metadata.

    Args:
        path: Location of the metadata file. Defaults to
            ``settings.training_metadata_path``.

    Returns:
        The training metadata dictionary.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
