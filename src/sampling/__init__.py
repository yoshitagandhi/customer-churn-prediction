"""Class imbalance handling and sampling experiment framework.

Analyzes class imbalance, compares interchangeable sampling
strategies against the training fold, and reports the results. This
package never trains or evaluates a model — it only characterizes
how each strategy affects dataset size and class balance, so
Milestone 6 can pick a strategy with that information in hand.

This package operates on an already-split training fold
(``features``, ``target``); it does not perform the train/test split
itself.

Typical usage::

    from src.sampling import run_sampling_analysis

    results = run_sampling_analysis(X_train, y_train)
"""

from typing import Any

import pandas as pd

from src.utils.exceptions import DataValidationError
from configs.config import settings
from configs.logging_config import get_logger
from src.sampling.experiment import (
    SamplingExperimentResult,
    compare_sampling_strategies,
    run_sampling_experiment,
)
from src.sampling.imbalance_analysis import analyze_class_distribution
from src.sampling.pipeline import (
    build_sampling_pipeline,
    fit_resample_training_data,
    transform_holdout_data,
)
from src.sampling.report import generate_sampling_report
from src.sampling.samplers import (
    SUPPORTED_STRATEGIES,
    compute_scale_pos_weight,
    get_class_weight_parameter,
    get_sampler,
    list_supported_samplers,
)

logger = get_logger(__name__)

__all__ = [
    "analyze_class_distribution",
    "get_sampler",
    "list_supported_samplers",
    "get_class_weight_parameter",
    "compute_scale_pos_weight",
    "build_sampling_pipeline",
    "fit_resample_training_data",
    "transform_holdout_data",
    "run_sampling_experiment",
    "compare_sampling_strategies",
    "SamplingExperimentResult",
    "generate_sampling_report",
    "run_sampling_analysis",
]


def run_sampling_analysis(
    features: pd.DataFrame,
    target: pd.Series,
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES,
) -> dict[str, Any]:
    """Run the full sampling analysis and comparison pipeline.

    Args:
        features: Training features (raw, pre-preprocessing). Must be
            the training fold only — never the full dataset or a
            validation/test split.
        target: Training labels, aligned with ``features``.
        strategies: Sampling strategies to compare. Defaults to every
            strategy this project supports.

    Returns:
        A dictionary with the original class distribution analysis,
        the list of per-strategy experiment results, and the paths of
        the generated reports.
    """
    
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

    logger.info("Starting sampling analysis.")
    
    class_distribution = analyze_class_distribution(target)
    experiment_results = compare_sampling_strategies(
        features, target, strategies, random_state=settings.random_seed
    )
    report_paths = generate_sampling_report(class_distribution, experiment_results)
    logger.info("Starting sampling analysis.")
    logger.info("Analyzing class distribution.")
    logger.info("Running sampling experiments.")
    logger.info("Generating sampling reports.")
    logger.info("Sampling analysis completed.")

    return {
        "class_distribution": class_distribution,
        "experiment_results": experiment_results,
        "report_paths": report_paths,
    }
