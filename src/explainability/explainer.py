"""SHAP explainer initialization.

Loads the serialized best model (Milestone 6/7) and initializes the
appropriate SHAP explainer for it. Kept separate from visualization
and reporting logic — this module only prepares the explainer and
computes raw SHAP values.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.models import load_model, load_training_metadata

logger = get_logger(__name__)


def load_explainer_inputs(
    model_path: Path = settings.best_model_path,
    metadata_path: Path = settings.training_metadata_path,
) -> dict[str, Any]:
    """Load the serialized best model and its training metadata.

    Args:
        model_path: Path to the serialized best model (the complete
            fitted preprocessing + [sampling] + model pipeline).
            Defaults to ``settings.best_model_path``.
        metadata_path: Path to the training metadata JSON. Defaults
            to ``settings.training_metadata_path``.

    Returns:
        A dictionary with the loaded pipeline and its metadata.
    """
    logger.info("Explainability started.")
    pipeline = load_model(model_path)
    metadata = load_training_metadata(metadata_path)
    logger.info("Model loaded.")
    return {"pipeline": pipeline, "metadata": metadata}


def get_processed_features(pipeline: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Transform raw features through the fitted preprocessing step only.

    SHAP explains the model in the space it actually operates on (the
    processed, numeric/encoded feature matrix) — not the raw,
    pre-preprocessing columns.

    Args:
        pipeline: The fitted preprocessing + [sampling] + model
            pipeline (sampling is automatically skipped during
            ``.transform()``).
        features: Raw feature data (pre-preprocessing).

    Returns:
        A DataFrame of processed features with meaningful column
        names, matching exactly what the model itself was fit on.
    """
    named_steps = pipeline.named_steps

    if "preprocessing" in named_steps:
        preprocessing_step = named_steps["preprocessing"]
        transformed = preprocessing_step.transform(features)
        column_transformer = preprocessing_step.named_steps["column_transformer"]
    else:
        # Current training pipelines store preprocessing as flattened steps.
        cleaned = named_steps["cleaning"].transform(features)
        engineered = named_steps["feature_engineering"].transform(cleaned)
        column_transformer = named_steps["column_transformer"]
        transformed = column_transformer.transform(engineered)

    feature_names = column_transformer.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=features.index)


def load_explainer(
    pipeline: Any,
    background_features: pd.DataFrame,
    max_background_samples: int = settings.shap_max_background_samples,
    random_state: int = settings.random_seed,
) -> Any:
    """Initialize a SHAP explainer for the pipeline's final model step.

    Uses SHAP's unified ``Explainer`` API, which automatically selects
    the appropriate algorithm (Tree, Linear, Kernel/Permutation, ...)
    based on the model type. The explainer wraps
    ``model.predict_proba`` (rather than the raw model object) so the
    same code path works uniformly across every registered model type
    — including ones without a specialized fast-path explainer.

    Args:
        pipeline: The fitted preprocessing + [sampling] + model
            pipeline.
        background_features: Processed feature data (see
            :func:`get_processed_features`) used as the background /
            reference distribution for the explainer.
        max_background_samples: Maximum number of background samples
            to use, for performance. Defaults to
            ``settings.shap_max_background_samples``.
        random_state: Random seed used when subsampling the
            background data. Defaults to ``settings.random_seed``.

    Returns:
        A fitted SHAP Explainer instance.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The 'shap' package is required for explainability. Install it via `pip install shap`."
        ) from exc

    model = pipeline.named_steps["model"]
    background_sample = background_features
    if len(background_features) > max_background_samples:
        background_sample = background_features.sample(
            n=max_background_samples, random_state=random_state
        )

    explainer = shap.Explainer(model.predict_proba, background_sample)
    logger.debug("SHAP explainer initialized for model type: %s.", type(model).__name__)
    return explainer


def compute_shap_values(explainer: Any, features: pd.DataFrame) -> Any:
    """Compute SHAP values for a set of processed feature rows.

    Args:
        explainer: A SHAP Explainer instance from :func:`load_explainer`.
        features: Processed feature data (the same space the
            explainer's background was built on).

    Returns:
        A SHAP Explanation object.
    """
    shap_values = explainer(features)
    logger.info("SHAP values computed.")
    return shap_values
