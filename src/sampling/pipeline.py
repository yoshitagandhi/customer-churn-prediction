"""Leakage-safe sampling pipeline.

Combines the Milestone 4 preprocessing pipeline with an interchangeable
sampling strategy using ``imblearn.pipeline.Pipeline`` — not
scikit-learn's ``Pipeline``, since only imbalanced-learn's version
knows how to skip the resampling step during ``.transform()``/
``.predict()`` calls. This is what makes it safe to reuse the same
pipeline object for training (resample) and inference (never
resample) without any special-casing by the caller.

Expected flow::

    Raw Dataset -> Preprocessing (Milestone 4) -> Sampling -> Model (Milestone 6)
"""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.utils.exceptions import DataValidationError
from configs.config import settings
from configs.logging_config import get_logger
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.sampling.samplers import get_sampler

logger = get_logger(__name__)


def build_sampling_pipeline(
    strategy: str = settings.default_sampling_strategy,
    preprocessor: SklearnPipeline | None = None,
    random_state: int = settings.random_seed,
) -> Any:
    """Build an unfitted preprocessing + sampling pipeline.

    Args:
        strategy: Sampling strategy identifier (see
            :data:`src.sampling.samplers.SUPPORTED_STRATEGIES`).
            Defaults to ``settings.default_sampling_strategy``.
        preprocessor: An unfitted preprocessing pipeline, as built by
            :func:`src.preprocessing.pipeline.build_preprocessing_pipeline`.
            A fresh one is built if not provided. Passing your own
            allows reusing an identically-configured preprocessor
            across multiple strategy comparisons.
        random_state: Random seed for reproducible resampling.
            Defaults to ``settings.random_seed``.

    Returns:
        An unfitted ``imblearn.pipeline.Pipeline`` with a
        "preprocessing" step and, for resampling strategies, a
        "sampling" step. Algorithm-level strategies ("class_weight",
        "scale_pos_weight") and "none" produce a pipeline with only
        the "preprocessing" step, since they have nothing to resample.

    Raises:
        ImportError: If a resampling strategy is requested but
            ``imbalanced-learn`` is not installed.
    """
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:
        raise ImportError(
            "The 'imbalanced-learn' package is required to build a sampling pipeline. "
            "Install it via `pip install imbalanced-learn`."
        ) from exc

    preprocessing_pipeline = (
        preprocessor if preprocessor is not None else build_preprocessing_pipeline()
    )

    sampler = get_sampler(strategy=strategy, random_state=random_state)

    # Flatten sklearn Pipeline into individual steps
    steps = list(preprocessing_pipeline.steps)

    if sampler is not None:
        steps.append(("sampling", sampler))

    return ImbPipeline(steps=steps)
        

def fit_resample_training_data(
    pipeline: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit the pipeline and resample the training fold in one leakage-safe call.

    This must only ever be called with the training fold. Validation,
    test, and inference data must instead go through
    :func:`transform_holdout_data`, which never resamples.

    Args:
        pipeline: An unfitted pipeline from :func:`build_sampling_pipeline`.
        features: Training features (raw, pre-preprocessing).
        target: Training labels.

    Returns:
        A tuple of (resampled feature DataFrame, resampled target
        Series). If the pipeline has no sampling step (baseline or
        algorithm-level strategies), the data is preprocessed but not
        resampled, and its size matches the input.
    """
    logger.info("Sampling strategy applied.")

    if "sampling" in pipeline.named_steps:
        processed_features, resampled_target = pipeline.fit_resample(features, target)
    else:
        # No sampler step (baseline "none" or an algorithm-level
        # strategy): behaves like a plain preprocessing transform,
        # and the target is returned unchanged.
        processed_features = pipeline.fit_transform(features, target)
        resampled_target = target

    feature_names = (
        pipeline.named_steps["preprocessing"]
        .named_steps["column_transformer"]
        .get_feature_names_out()
    )
    resampled_features = pd.DataFrame(processed_features, columns=feature_names)

    logger.info(
        "Resampling completed: %d -> %d row(s).", len(features), len(resampled_features)
    )
    return resampled_features, pd.Series(resampled_target, name=target.name)


def transform_holdout_data(pipeline: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Transform validation/test/inference data using only the fitted preprocessing step.

    Never resamples — safe to call on any data that must not be
    altered in size or composition.

    Args:
        pipeline: A fitted pipeline from :func:`build_sampling_pipeline`
            (already fit via :func:`fit_resample_training_data`).
        features: Data to transform (validation, test, or inference).

    Returns:
        A DataFrame of processed features, with the same number of
        rows as ``features``.
    """
    preprocessing_step = pipeline.named_steps["preprocessing"]
    transformed_array = preprocessing_step.transform(features)
    feature_names = preprocessing_step.named_steps["column_transformer"].get_feature_names_out()
    return pd.DataFrame(transformed_array, columns=feature_names, index=features.index)
