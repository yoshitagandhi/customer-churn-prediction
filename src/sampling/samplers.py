"""Reusable sampling strategy implementations.

Provides a single factory, :func:`get_sampler`, for every supported
sampling strategy, plus utilities for the two algorithm-level
balancing techniques (``class_weight`` and ``scale_pos_weight``) that
do not resample data at all — they configure the model itself, which
happens in Milestone 6.

The actual resampling algorithms (SMOTE, BorderlineSMOTE, SMOTETomek,
RandomOverSampler) come from the ``imbalanced-learn`` package. That
import is deferred until a resampling strategy is actually requested,
so the rest of this project (and this module's own
non-resampling utilities) remain usable even in an environment where
``imbalanced-learn`` has not been installed yet.
"""

from typing import Any, Final

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)

# Strategies that do not resample data; they configure a model's
# fitting behavior instead (handled in Milestone 6).
ALGORITHM_LEVEL_STRATEGIES: Final[tuple[str, ...]] = ("class_weight", "scale_pos_weight")

# Strategies that produce a resampled (X, y) via imbalanced-learn.
RESAMPLING_STRATEGIES: Final[tuple[str, ...]] = (
    "random_oversample",
    "smote",
    "borderline_smote",
    "smote_tomek",
)

SUPPORTED_STRATEGIES: Final[tuple[str, ...]] = (
    ("none",) + RESAMPLING_STRATEGIES + ALGORITHM_LEVEL_STRATEGIES
)


def list_supported_samplers() -> tuple[str, ...]:
    """Return every sampling strategy identifier this module supports.

    Returns:
        A tuple of strategy identifiers.
    """
    return SUPPORTED_STRATEGIES


def get_sampler(
    strategy: str,
    random_state: int = settings.random_seed,
    k_neighbors: int = settings.smote_k_neighbors,
) -> Any | None:
    """Build a fresh, unfitted sampler instance for a given strategy.

    Args:
        strategy: One of :data:`SUPPORTED_STRATEGIES`.
        random_state: Random seed for reproducible resampling.
            Defaults to ``settings.random_seed``.
        k_neighbors: Number of nearest neighbors used by SMOTE-family
            samplers. Defaults to ``settings.smote_k_neighbors``.

    Returns:
        An unfitted imbalanced-learn sampler instance for resampling
        strategies, or None for "none" and the algorithm-level
        strategies (which have no resampling component).

    Raises:
        ConfigurationError: If ``strategy`` is not supported.
        ImportError: If a resampling strategy is requested but
            ``imbalanced-learn`` is not installed.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ConfigurationError(
            f"Unsupported sampling strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}"
        )

    if strategy == "none" or strategy in ALGORITHM_LEVEL_STRATEGIES:
        return None

    return _build_resampler(strategy, random_state, k_neighbors)


def _build_resampler(strategy: str, random_state: int, k_neighbors: int) -> Any:
    """Instantiate the imbalanced-learn resampler for a resampling strategy.

    Args:
        strategy: One of :data:`RESAMPLING_STRATEGIES`.
        random_state: Random seed for reproducible resampling.
        k_neighbors: Number of nearest neighbors for SMOTE-family
            samplers.

    Returns:
        An unfitted imbalanced-learn sampler instance.

    Raises:
        ImportError: If ``imbalanced-learn`` is not installed.
    """
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
    except ImportError as exc:
        raise ImportError(
            "The 'imbalanced-learn' package is required for sampling strategy "
            f"'{strategy}'. Install it via `pip install imbalanced-learn`."
        ) from exc

    if strategy == "random_oversample":
        return RandomOverSampler(random_state=random_state)
    if strategy == "smote":
        return SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    if strategy == "borderline_smote":
        return BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
    if strategy == "smote_tomek":
        inner_smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        return SMOTETomek(random_state=random_state, smote=inner_smote)

    # Unreachable given the SUPPORTED_STRATEGIES check in get_sampler().
    raise ConfigurationError(f"No resampler implementation for strategy '{strategy}'.")


def get_class_weight_parameter() -> str:
    """Return the standard scikit-learn ``class_weight`` value for balanced weighting.

    Returns:
        The string "balanced", suitable for any scikit-learn
        classifier's ``class_weight`` parameter.
    """
    return "balanced"


def compute_scale_pos_weight(target: pd.Series, positive_label: str) -> float:
    """Compute the ``scale_pos_weight`` ratio for XGBoost-style models.

    Defined as the negative-class count divided by the positive-class
    count, which is the value XGBoost's ``scale_pos_weight`` parameter
    expects to up-weight the minority (positive) class.

    Args:
        target: The target Series to analyze.
        positive_label: The label representing the positive
            (typically minority) class, e.g. "Yes" for churn.

    Returns:
        The computed scale_pos_weight ratio.

    Raises:
        ConfigurationError: If ``positive_label`` is not present in
            ``target``, or if it has zero occurrences.
    """
    counts = target.value_counts()
    if positive_label not in counts.index:
        raise ConfigurationError(
            f"positive_label '{positive_label}' not found in target values: "
            f"{counts.index.tolist()}"
        )

    positive_count = int(counts[positive_label])
    negative_count = int(counts.sum() - positive_count)
    if positive_count == 0:
        raise ConfigurationError("Cannot compute scale_pos_weight: positive class has 0 samples.")

    return round(float(negative_count) / float(positive_count), 4)
