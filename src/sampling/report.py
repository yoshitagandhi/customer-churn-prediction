"""Sampling experiment report generation.

Turns the output of :mod:`src.sampling.imbalance_analysis` and
:mod:`src.sampling.experiment` into ``sampling_report.md`` (human
readable) and ``sampling_summary.json`` (machine readable).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configs.config import settings
from configs.logging_config import get_logger
from src.sampling.experiment import SamplingExperimentResult
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)


def generate_sampling_report(
    class_distribution: dict[str, Any],
    experiment_results: list[SamplingExperimentResult],
    json_path: Path = settings.sampling_summary_path,
    markdown_path: Path = settings.sampling_report_path,
) -> dict[str, Path]:
    """Generate the JSON and Markdown sampling experiment reports.

    Args:
        class_distribution: Output of
            :func:`src.sampling.imbalance_analysis.analyze_class_distribution`
            for the original (pre-sampling) training target.
        experiment_results: Output of
            :func:`src.sampling.experiment.compare_sampling_strategies`.
        json_path: Destination for the JSON summary. Defaults to
            ``settings.sampling_summary_path``.
        markdown_path: Destination for the Markdown report. Defaults
            to ``settings.sampling_report_path``.

    Returns:
        A mapping with keys "json" and "markdown" pointing to the
        generated file paths.
    """
    if not class_distribution:
        raise DataValidationError(
            "Class distribution cannot be empty."
        )

    if experiment_results is None:
        raise DataValidationError(
            "Experiment results cannot be None."
        )

    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "original_class_distribution": class_distribution,
        "experiments": [result.to_dict() for result in experiment_results],
        "recommendations": _generate_recommendations(class_distribution, experiment_results),
        "report_version": "1.0",
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report_data), encoding="utf-8")

    logger.info("Writing JSON sampling report.")
    logger.info("Writing Markdown sampling report.")
    logger.info("Sampling report generation completed.")
    return {"json": json_path, "markdown": markdown_path}


def _generate_recommendations(
    class_distribution: dict[str, Any], experiment_results: list[SamplingExperimentResult]
) -> list[str]:
    """Derive recommendations grounded only in the observed experiment results.

    Args:
        class_distribution: Original class distribution analysis.
        experiment_results: Results from every strategy tested.

    Returns:
        A list of recommendation strings. Recommendations describe
        resampling mechanics only (dataset size, resulting balance) —
        not predicted model performance, which this milestone does
        not evaluate.
    """
    recommendations: list[str] = []

    if not class_distribution["sampling_recommended"]:
        recommendations.append(
            f"The training data's imbalance ratio ({class_distribution['imbalance_ratio']}:1) "
            "is mild; algorithm-level balancing (class_weight or scale_pos_weight) may be "
            "sufficient without resampling."
        )
    else:
        recommendations.append(
            f"The training data's imbalance ratio ({class_distribution['imbalance_ratio']}:1) "
            "suggests resampling is worth evaluating alongside algorithm-level balancing."
        )

    excluded_strategies = ("none", "class_weight", "scale_pos_weight")
    resampling_results = [r for r in experiment_results if r.strategy not in excluded_strategies]
    if resampling_results:
        most_balanced = min(resampling_results, key=lambda r: r.sampled_imbalance_ratio)
        recommendations.append(
            f"Among tested resampling strategies, '{most_balanced.strategy}' produced the most "
            f"balanced training set ({most_balanced.sampled_imbalance_ratio}:1, "
            f"{most_balanced.sampled_size} rows, up from {most_balanced.original_size})."
        )

    recommendations.append(
        "These results describe dataset size and balance only. Which strategy yields the best "
        "model performance must still be validated empirically during model training and "
        "evaluation (Milestone 6)."
    )
    return recommendations


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the sampling report as a Markdown document.

    Args:
        report: The structured report data built in
            :func:`generate_sampling_report`.

    Returns:
        The complete Markdown document as a single string.
    """
    sections = [
        f"# Sampling Experiment Report — {settings.project_name}",
        f"\nGenerated at: {report['generated_at']}\n",
        _render_original_distribution_section(report["original_class_distribution"]),
        _render_comparison_table_section(report["experiments"]),
        _render_recommendations_section(report["recommendations"]),
    ]
    return "\n".join(sections)


def _render_original_distribution_section(distribution: dict[str, Any]) -> str:
    """Render the "Original Class Distribution" Markdown section."""
    lines = [
        "## Original Class Distribution\n",
        f"- Total samples: {distribution['total_samples']}",
        f"- Majority class: {distribution['majority_class']} ({distribution['majority_count']})",
        f"- Minority class: {distribution['minority_class']} ({distribution['minority_count']})",
        f"- Imbalance ratio: {distribution['imbalance_ratio']}:1",
        f"- Sampling recommended: {distribution['sampling_recommended']}",
        "",
        distribution["summary"],
    ]
    return "\n".join(lines) + "\n"


def _render_comparison_table_section(experiments: list[dict[str, Any]]) -> str:
    """Render the strategy comparison Markdown table."""
    if not experiments:
        return "## Strategy Comparison\n\nNo experiments were run.\n"

    lines = [
        "## Strategy Comparison\n",
        "| Strategy | Original Size | Sampled Size | Original Ratio | Sampled Ratio | Time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for experiment in experiments:
        lines.append(
            f"| {experiment['strategy']} | {experiment['original_size']} | "
            f"{experiment['sampled_size']} | {experiment['original_imbalance_ratio']}:1 | "
            f"{experiment['sampled_imbalance_ratio']}:1 | {experiment['execution_time_seconds']} |"
        )

    lines.append("\n### Notes\n")
    lines.extend(
        f"- **{experiment['strategy']}**: {experiment['notes']}" for experiment in experiments
    )
    return "\n".join(lines) + "\n"


def _render_recommendations_section(recommendations: list[str]) -> str:
    """Render the "Recommendations" Markdown section."""
    lines = ["## Recommendations\n"]
    lines.extend(f"- {recommendation}" for recommendation in recommendations)
    return "\n".join(lines) + "\n"
