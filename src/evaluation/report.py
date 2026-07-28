"""Evaluation report generation.

Turns the outputs of :mod:`src.evaluation.metrics`,
:mod:`src.evaluation.comparison`, and :mod:`src.evaluation.calibration`
into three artifacts: ``evaluation_metrics.json`` (machine-readable),
``model_ranking.csv``, and ``evaluation_report.md`` (human-readable).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


def generate_evaluation_report(
    comparison_frame: pd.DataFrame,
    best_model_info: dict[str, Any],
    metrics_by_model: dict[str, dict[str, Any]],
    figure_paths: dict[str, Any],
    calibration_result: dict[str, Any],
    metrics_path: Path = settings.evaluation_metrics_path,
    ranking_path: Path = settings.model_ranking_path,
    report_path: Path = settings.evaluation_report_path,
) -> dict[str, Path]:
    """Generate the JSON, CSV, and Markdown evaluation artifacts.

    Args:
        comparison_frame: Output of
            :func:`src.evaluation.comparison.compare_models`.
        best_model_info: Output of
            :func:`src.evaluation.comparison.identify_best_model`.
        metrics_by_model: Mapping of model name to its metrics
            dictionary.
        figure_paths: Mapping of figure identifiers to their saved
            file paths, for reference in the report.
        calibration_result: Output of
            :func:`src.evaluation.calibration.generate_calibration_curve`
            for the best model.
        metrics_path: Destination for the JSON metrics file. Defaults
            to ``settings.evaluation_metrics_path``.
        ranking_path: Destination for the CSV ranking file. Defaults
            to ``settings.model_ranking_path``.
        report_path: Destination for the Markdown report. Defaults to
            ``settings.evaluation_report_path``.

    Returns:
        A mapping with keys "metrics_json", "ranking_csv", and
        "report_markdown" pointing to the generated file paths.
    """
    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics_by_model": metrics_by_model,
        "best_model": best_model_info,
        "calibration": calibration_result,
        "figures": figure_paths,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    comparison_frame.to_csv(ranking_path, index=False)
    report_path.write_text(
        _render_markdown(comparison_frame, best_model_info, calibration_result, figure_paths),
        encoding="utf-8",
    )

    logger.info("Reports created: %s, %s, %s.", metrics_path, ranking_path, report_path)
    return {
        "metrics_json": metrics_path,
        "ranking_csv": ranking_path,
        "report_markdown": report_path,
    }


def _render_markdown(
    comparison_frame: pd.DataFrame,
    best_model_info: dict[str, Any],
    calibration_result: dict[str, Any],
    figure_paths: dict[str, Any],
) -> str:
    """Render the full evaluation report as a Markdown document.

    Args:
        comparison_frame: Output of
            :func:`src.evaluation.comparison.compare_models`.
        best_model_info: Output of
            :func:`src.evaluation.comparison.identify_best_model`.
        calibration_result: Calibration results for the best model.
        figure_paths: Mapping of figure identifiers to saved paths.

    Returns:
        The complete Markdown document as a single string.
    """
    strengths, weaknesses = _generate_strengths_and_weaknesses(best_model_info, calibration_result)
    recommendations = _generate_recommendations(best_model_info, calibration_result)

    sections = [
        f"# Model Evaluation Report — {settings.project_name}",
        f"\nGenerated at: {datetime.now(UTC).isoformat()}\n",
        _render_overall_summary(comparison_frame, best_model_info),
        _render_metrics_table(comparison_frame),
        _render_best_model_section(best_model_info, calibration_result),
        _render_bulleted_section("Strengths", strengths),
        _render_bulleted_section("Weaknesses", weaknesses),
        _render_bulleted_section("Recommendations", recommendations),
        _render_figures_section(figure_paths),
    ]
    return "\n".join(sections)


def _render_overall_summary(comparison_frame: pd.DataFrame, best_model_info: dict[str, Any]) -> str:
    """Render the "Overall Summary" Markdown section."""
    return (
        "## Overall Summary\n\n"
        f"{len(comparison_frame)} model(s) were evaluated on the held-out test set using "
        f"ROC-AUC as the primary ranking metric (PR-AUC, F1, and Recall as tie-breakers). "
        f"The best-performing model was **{best_model_info['model_name']}** "
        f"(sampling strategy: {best_model_info.get('sampling_strategy', 'n/a')}).\n"
    )


def _render_metrics_table(comparison_frame: pd.DataFrame) -> str:
    """Render the full metrics comparison table as Markdown."""
    display_columns = [
        col
        for col in (
            "model_name",
            "sampling_strategy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "training_time_seconds",
        )
        if col in comparison_frame.columns
    ]
    lines = [
        "## Metrics Table\n",
        "| " + " | ".join(display_columns) + " |",
        "|" + "---|" * len(display_columns),
    ]
    for _, row in comparison_frame[display_columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display_columns) + " |")
    return "\n".join(lines) + "\n"


def _render_best_model_section(
    best_model_info: dict[str, Any], calibration_result: dict[str, Any]
) -> str:
    """Render the "Best Model" Markdown section."""
    lines = [
        "## Best Model\n",
        f"- Model: **{best_model_info['model_name']}**",
        f"- Sampling strategy: {best_model_info.get('sampling_strategy', 'n/a')}",
        f"- ROC-AUC: {best_model_info.get('roc_auc')}",
        f"- PR-AUC: {best_model_info.get('pr_auc')}",
        f"- Precision: {best_model_info.get('precision')}",
        f"- Recall: {best_model_info.get('recall')}",
        f"- F1: {best_model_info.get('f1')}",
        f"- Brier score (calibration): {calibration_result.get('brier_score')}",
        f"- Selection reason: {best_model_info.get('selection_reason', 'n/a')}",
    ]
    return "\n".join(lines) + "\n"


def _render_bulleted_section(title: str, items: list[str]) -> str:
    """Render a simple bulleted Markdown section."""
    lines = [f"## {title}\n"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines) + "\n"


def _render_figures_section(figure_paths: dict[str, Any]) -> str:
    """Render the "Figures Generated" Markdown section."""
    lines = ["## Figures Generated\n"]
    for name, path in figure_paths.items():
        lines.append(f"- **{name}**: `{path}`")
    return "\n".join(lines) + "\n"


def _generate_strengths_and_weaknesses(
    best_model_info: dict[str, Any], calibration_result: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Derive strengths/weaknesses statements grounded in the best model's own metrics.

    Args:
        best_model_info: Output of
            :func:`src.evaluation.comparison.identify_best_model`.
        calibration_result: Calibration results for the best model.

    Returns:
        A tuple of (strengths, weaknesses) lists.
    """
    strengths: list[str] = []
    weaknesses: list[str] = []

    roc_auc = best_model_info.get("roc_auc", 0.0)
    if roc_auc >= 0.8:
        strengths.append(f"Strong discriminative ability (ROC-AUC={roc_auc}).")
    elif roc_auc < 0.7:
        weaknesses.append(f"Limited discriminative ability (ROC-AUC={roc_auc}).")

    precision = best_model_info.get("precision", 0.0)
    if precision >= 0.7:
        strengths.append(f"High precision ({precision}): few false churn alarms.")
    elif precision < 0.5:
        weaknesses.append(f"Low precision ({precision}): may over-predict churn.")

    recall = best_model_info.get("recall", 0.0)
    if recall >= 0.7:
        strengths.append(f"High recall ({recall}): captures most actual churners.")
    elif recall < 0.5:
        weaknesses.append(f"Low recall ({recall}): may miss many actual churners.")

    brier_score = calibration_result.get("brier_score")
    if brier_score is not None:
        if brier_score <= 0.15:
            strengths.append(
                f"Reasonably well-calibrated probabilities (Brier score={brier_score})."
            )
        else:
            weaknesses.append(
                f"Predicted probabilities may be poorly calibrated (Brier score={brier_score})."
            )

    if not strengths:
        strengths.append("No metric exceeded the threshold used to flag a clear strength.")
    if not weaknesses:
        weaknesses.append("No metric fell below the threshold used to flag a clear weakness.")

    return strengths, weaknesses


def _generate_recommendations(
    best_model_info: dict[str, Any], calibration_result: dict[str, Any]
) -> list[str]:
    """Derive next-step recommendations grounded in the best model's own metrics.

    Args:
        best_model_info: Output of
            :func:`src.evaluation.comparison.identify_best_model`.
        calibration_result: Calibration results for the best model.

    Returns:
        A list of recommendation strings.
    """
    recommendations: list[str] = [
        f"Proceed to SHAP explainability (Milestone 8) to understand which features drive "
        f"'{best_model_info['model_name']}' predictions.",
    ]

    if best_model_info.get("recall", 1.0) < 0.7 or best_model_info.get("precision", 1.0) < 0.7:
        recommendations.append(
            "Consider threshold optimization (Milestone 9) to adjust the precision/recall "
            "trade-off according to the business cost of false negatives vs. false positives."
        )

    brier_score = calibration_result.get("brier_score")
    if brier_score is not None and brier_score > 0.15:
        recommendations.append(
            "Consider probability calibration (e.g., Platt scaling or isotonic regression) "
            "before using raw predicted probabilities for business decisions."
        )

    recommendations.append(
        "This model was selected automatically from evaluation metrics on the held-out test "
        "set; no winner was hardcoded."
    )
    return recommendations
