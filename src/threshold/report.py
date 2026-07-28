"""Threshold optimization report generation.

Turns the outputs of :mod:`src.threshold.optimizer`,
:mod:`src.threshold.decision_engine`, and the threshold evaluation
table into three artifacts: ``threshold_report.md``,
``threshold_metrics.csv``, and ``business_decision_summary.json``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


def generate_threshold_report(
    optimization_result: dict[str, Any],
    decision_table: pd.DataFrame,
    figure_paths: dict[str, Any],
    report_path: Path = settings.threshold_report_path,
    metrics_csv_path: Path = settings.threshold_metrics_csv_path,
    decision_summary_path: Path = settings.business_decision_summary_path,
) -> dict[str, Path]:
    """Generate the Markdown, CSV, and JSON threshold optimization artifacts.

    Args:
        optimization_result: Output of
            :func:`src.threshold.optimizer.optimize_threshold`.
        decision_table: Output of
            :func:`src.threshold.decision_engine.generate_recommendations`.
        figure_paths: Mapping of figure identifiers to their saved
            file paths, for reference in the report.
        report_path: Destination for the Markdown report. Defaults to
            ``settings.threshold_report_path``.
        metrics_csv_path: Destination for the full threshold
            evaluation table CSV. Defaults to
            ``settings.threshold_metrics_csv_path``.
        decision_summary_path: Destination for the business decision
            summary JSON. Defaults to
            ``settings.business_decision_summary_path``.

    Returns:
        A mapping with keys "report_markdown", "metrics_csv", and
        "decision_summary_json" pointing to the generated file paths.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    decision_summary_path.parent.mkdir(parents=True, exist_ok=True)

    evaluation_table = optimization_result["evaluation_table"]
    evaluation_table.to_csv(metrics_csv_path, index=False)

    risk_level_counts = decision_table["risk_level"].value_counts().to_dict()
    decision_summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "optimal_threshold": optimization_result["optimal_threshold"],
        "objective": optimization_result["objective"],
        "metrics_at_optimal_threshold": optimization_result["metrics_at_optimal_threshold"],
        "risk_level_counts": risk_level_counts,
        "total_customers_evaluated": int(len(decision_table)),
    }
    decision_summary_path.write_text(
        json.dumps(decision_summary, indent=2, default=str), encoding="utf-8"
    )

    report_path.write_text(
        _render_markdown(optimization_result, decision_summary, figure_paths), encoding="utf-8"
    )

    logger.info(
        "Reports generated: %s, %s, %s.", report_path, metrics_csv_path, decision_summary_path
    )
    return {
        "report_markdown": report_path,
        "metrics_csv": metrics_csv_path,
        "decision_summary_json": decision_summary_path,
    }


def _render_markdown(
    optimization_result: dict[str, Any],
    decision_summary: dict[str, Any],
    figure_paths: dict[str, Any],
) -> str:
    """Render the full threshold optimization report as a Markdown document.

    Args:
        optimization_result: Output of
            :func:`src.threshold.optimizer.optimize_threshold`.
        decision_summary: The structured decision summary assembled
            in :func:`generate_threshold_report`.
        figure_paths: Mapping of figure identifiers to saved paths.

    Returns:
        The complete Markdown document as a single string.
    """
    sections = [
        f"# Threshold Optimization Report — {settings.project_name}",
        f"\nGenerated at: {datetime.now(UTC).isoformat()}\n",
        _render_selected_threshold_section(optimization_result),
        _render_comparison_highlights_section(optimization_result["evaluation_table"]),
        _render_business_cost_section(optimization_result["metrics_at_optimal_threshold"]),
        _render_business_impact_section(decision_summary),
        _render_figures_section(figure_paths),
        _render_deployment_recommendation_section(optimization_result),
    ]
    return "\n".join(sections)


def _render_selected_threshold_section(optimization_result: dict[str, Any]) -> str:
    """Render the "Selected Threshold" Markdown section."""
    metrics = optimization_result["metrics_at_optimal_threshold"]
    return (
        "## Selected Threshold\n\n"
        f"- Optimization objective: **{optimization_result['objective']}**\n"
        f"- Optimal threshold: **{optimization_result['optimal_threshold']}**\n"
        f"- Precision: {metrics['precision']}\n"
        f"- Recall: {metrics['recall']}\n"
        f"- F1: {metrics['f1']}\n"
    )


def _render_comparison_highlights_section(evaluation_table: pd.DataFrame) -> str:
    """Render the "Comparison Highlights" Markdown section (best-of-each-metric thresholds)."""
    lowest_cost_row = evaluation_table.loc[evaluation_table["business_cost"].idxmin()]
    highest_f1_row = evaluation_table.loc[evaluation_table["f1"].idxmax()]
    highest_recall_row = evaluation_table.loc[evaluation_table["recall"].idxmax()]

    return (
        "## Comparison Highlights\n\n"
        f"- Lowest business cost: threshold={lowest_cost_row['threshold']} "
        f"(cost={lowest_cost_row['business_cost']})\n"
        f"- Highest F1: threshold={highest_f1_row['threshold']} (F1={highest_f1_row['f1']})\n"
        f"- Highest Recall: threshold={highest_recall_row['threshold']} "
        f"(Recall={highest_recall_row['recall']})\n"
    )


def _render_business_cost_section(metrics: dict[str, Any]) -> str:
    """Render the "Business Cost Summary" Markdown section."""
    return (
        "## Business Cost Summary\n\n"
        f"- False positive cost: {metrics['false_positive_cost']}\n"
        f"- False negative cost: {metrics['false_negative_cost']}\n"
        f"- Retention campaign cost: {metrics['retention_campaign_cost_total']}\n"
        f"- Expected avoided churn: {metrics['expected_avoided_churn']} customer(s)\n"
        f"- Estimated savings: {metrics['estimated_savings']}\n"
        f"- Net business cost: {metrics['business_cost']}\n"
    )


def _render_business_impact_section(decision_summary: dict[str, Any]) -> str:
    """Render the "Expected Business Impact" Markdown section."""
    lines = [
        "## Expected Business Impact\n",
        f"- Total customers evaluated: {decision_summary['total_customers_evaluated']}",
        "- Risk level distribution:",
    ]
    lines.extend(
        f"  - {level}: {count}" for level, count in decision_summary["risk_level_counts"].items()
    )
    return "\n".join(lines) + "\n"


def _render_figures_section(figure_paths: dict[str, Any]) -> str:
    """Render the "Figures Generated" Markdown section."""
    lines = ["## Figures Generated\n"]
    lines.extend(f"- **{name}**: `{path}`" for name, path in figure_paths.items())
    return "\n".join(lines) + "\n"


def _render_deployment_recommendation_section(optimization_result: dict[str, Any]) -> str:
    """Render the "Deployment Recommendation" Markdown section."""
    return (
        "## Deployment Recommendation\n\n"
        f"Use threshold **{optimization_result['optimal_threshold']}** "
        f"(selected via the '{optimization_result['objective']}' objective) as the default "
        "classification threshold for inference and the Streamlit application, replacing "
        "the arbitrary 0.50 default. Revisit this threshold if business cost parameters "
        "change or the model is retrained.\n"
    )
