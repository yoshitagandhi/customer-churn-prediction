"""Sampling strategy comparison framework.

Runs every supported sampling strategy against the same training
fold and records how dataset size and class distribution change. No
model is trained or evaluated here — that is Milestone 6's
responsibility; this module only characterizes the resampling effect
itself so a strategy can be chosen with that information in hand.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.utils.exceptions import DataValidationError
from configs.config import settings
from configs.logging_config import get_logger
from src.sampling.imbalance_analysis import analyze_class_distribution
from src.sampling.pipeline import build_sampling_pipeline, fit_resample_training_data
from src.sampling.samplers import ALGORITHM_LEVEL_STRATEGIES, SUPPORTED_STRATEGIES

logger = get_logger(__name__)

from typing import Final

_ALGORITHM_LEVEL_NOTES: Final[dict[str, str]] = {
    "class_weight": (
        "No resampling performed. Configure the Milestone 6 classifier with "
        "class_weight='balanced' instead of altering the training data."
    ),
    "scale_pos_weight": (
        "No resampling performed. Pass the computed scale_pos_weight ratio "
        "directly to the Milestone 6 XGBoost model instead of altering the training data."
    ),
}


@dataclass
class SamplingExperimentResult:
    """Outcome of running a single sampling strategy against the training fold.

    Attributes:
        strategy: Sampling strategy identifier.
        original_size: Number of training rows before sampling.
        sampled_size: Number of training rows after sampling.
        original_class_counts: Class counts before sampling.
        sampled_class_counts: Class counts after sampling.
        original_imbalance_ratio: Majority:minority ratio before
            sampling.
        sampled_imbalance_ratio: Majority:minority ratio after
            sampling.
        execution_time_seconds: How long the strategy took to run.
        notes: Human-readable context about this strategy's result.
    """

    strategy: str
    original_size: int
    sampled_size: int
    original_class_counts: dict[str, int]
    sampled_class_counts: dict[str, int]
    original_imbalance_ratio: float
    sampled_imbalance_ratio: float
    execution_time_seconds: float
    notes: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dictionary for reporting/serialization."""
        return asdict(self)


def run_sampling_experiment(
    features: pd.DataFrame,
    target: pd.Series,
    strategy: str,
    random_state: int = settings.random_seed,
) -> SamplingExperimentResult:
    
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
            "Features and target must have identical lengths."
    )
    
    """Run a single sampling strategy against the training fold and record its effect.

    Args:
        features: Training features (raw, pre-preprocessing).
        target: Training labels.
        strategy: Sampling strategy identifier. See
            :data:`src.sampling.samplers.SUPPORTED_STRATEGIES`.
        random_state: Random seed for reproducible resampling.
            Defaults to ``settings.random_seed``.

    Returns:
        A :class:`SamplingExperimentResult` describing the strategy's
        effect on dataset size and class distribution.
    """
    original_analysis = analyze_class_distribution(target)
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported sampling strategy: {strategy}"
    )
    
    start_time = time.perf_counter()

    if strategy == "none" or strategy in ALGORITHM_LEVEL_STRATEGIES:
        sampled_target = target
        notes = _ALGORITHM_LEVEL_NOTES.get(strategy, "Baseline: no sampling applied.")
    else:
        pipeline = build_sampling_pipeline(strategy=strategy, random_state=random_state)
        _, sampled_target = fit_resample_training_data(pipeline, features, target)
        notes = f"Resampling applied via imbalanced-learn's '{strategy}' strategy."

    execution_time = round(time.perf_counter() - start_time, 4)
    sampled_analysis = analyze_class_distribution(sampled_target)

    return SamplingExperimentResult(
        strategy=strategy,
        original_size=original_analysis["total_samples"],
        sampled_size=sampled_analysis["total_samples"],
        original_class_counts=original_analysis["class_counts"],
        sampled_class_counts=sampled_analysis["class_counts"],
        original_imbalance_ratio=original_analysis["imbalance_ratio"],
        sampled_imbalance_ratio=sampled_analysis["imbalance_ratio"],
        execution_time_seconds=execution_time,
        notes=notes,
    )


def compare_sampling_strategies(
    features: pd.DataFrame,
    target: pd.Series,
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES,
    random_state: int = settings.random_seed,
) -> list[SamplingExperimentResult]:
    """Run every requested sampling strategy and collect their results.

    Args:
        features: Training features (raw, pre-preprocessing).
        target: Training labels.
        strategies: Strategy identifiers to compare. Defaults to
            every strategy this project supports.
        random_state: Random seed for reproducible resampling.
            Defaults to ``settings.random_seed``.

    Returns:
        A list of :class:`SamplingExperimentResult`, one per strategy,
        in the same order as ``strategies``.
    """
    results: list[SamplingExperimentResult] = []
    for strategy in strategies:
        try:
            result = run_sampling_experiment(features, target, strategy, random_state)
        except Exception as exc:
            logger.exception(
                "Sampling strategy '%s' failed.",
                strategy,
            )
            continue
        
        results.append(result)
        logger.info("Experiment completed: strategy='%s'.", strategy)

    return results
