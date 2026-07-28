"""Automated EDA report generation.

Combines the outputs of :mod:`src.analysis.statistics` and
:mod:`src.analysis.business_insights` into three artifacts under the
reports directory: ``eda_statistics.json`` (machine-readable),
``eda_summary.md`` (human-readable overview), and
``business_insights.md`` (insights and recommendations only).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)

_JSON_FILENAME = "eda_statistics.json"
_SUMMARY_MARKDOWN_FILENAME = "eda_summary.md"
_INSIGHTS_MARKDOWN_FILENAME = "business_insights.md"


def generate_eda_report(
    dataset_statistics: dict[str, Any],
    numerical_statistics: dict[str, Any],
    categorical_statistics: dict[str, Any],
    target_statistics: dict[str, Any],
    correlation_summary: dict[str, Any],
    business_insights: dict[str, Any],
    figure_paths: dict[str, Any],
    output_dir: Path = settings.report_dir,
) -> dict[str, Path]:
    """Generate the EDA JSON report, summary Markdown, and insights Markdown.

    Args:
        dataset_statistics: Output of ``compute_dataset_statistics``.
        numerical_statistics: Output of ``compute_numerical_statistics``.
        categorical_statistics: Output of ``compute_categorical_statistics``.
        target_statistics: Output of ``compute_target_statistics``.
        correlation_summary: Output of ``get_top_correlations``.
        business_insights: Output of ``generate_business_insights``.
        figure_paths: Mapping of figure identifiers to their saved
            file paths, for reference in the report.
        output_dir: Directory reports are written to. Defaults to
            ``settings.report_dir``.

    Returns:
        A mapping with keys "json", "summary_markdown", and
        "insights_markdown" pointing to the generated file paths.
    """
    logger.info("Generating EDA report.")
    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_statistics": dataset_statistics,
        "numerical_statistics": numerical_statistics,
        "categorical_statistics": categorical_statistics,
        "target_statistics": target_statistics,
        "correlation_summary": correlation_summary,
        "business_insights": business_insights,
        "figures": figure_paths,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / _JSON_FILENAME
    summary_path = output_dir / _SUMMARY_MARKDOWN_FILENAME
    insights_path = output_dir / _INSIGHTS_MARKDOWN_FILENAME

    json_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    summary_path.write_text(_render_summary_markdown(report_data), encoding="utf-8")
    insights_path.write_text(_render_insights_markdown(business_insights), encoding="utf-8")

    logger.info("EDA reports saved: %s, %s, %s", json_path, summary_path, insights_path)
    return {"json": json_path, "summary_markdown": summary_path, "insights_markdown": insights_path}


def _render_summary_markdown(report: dict[str, Any]) -> str:
    """Render the full EDA summary as a Markdown document.

    Args:
        report: The structured report data assembled in
            ``generate_eda_report``.

    Returns:
        The complete Markdown document as a single string.
    """
    sections = [
        f"# EDA Summary — {settings.project_name}",
        f"\nGenerated at: {report['generated_at']}\n",
        _render_dataset_overview(report["dataset_statistics"], report["target_statistics"]),
        _render_numerical_summary(report["numerical_statistics"]),
        _render_categorical_summary(report["categorical_statistics"]),
        _render_correlation_summary(report["correlation_summary"]),
        _render_key_observations(report["business_insights"]),
        (
            "## Business Insights\n\n"
            "See `business_insights.md` for the full business insights and recommendations.\n"
        ),
    ]
    return "\n".join(sections)


def _render_dataset_overview(
    dataset_statistics: dict[str, Any], target_statistics: dict[str, Any]
) -> str:
    """Render the "Dataset Overview" and "Target Analysis" Markdown sections."""
    lines = [
        "## Dataset Overview\n",
        f"- Rows: {dataset_statistics['total_rows']}",
        f"- Columns: {dataset_statistics['total_columns']}",
        f"- Memory usage: {dataset_statistics['memory_usage_mb']} MB",
        "",
        "## Target Analysis\n",
        f"- Target column: `{target_statistics['target_column']}`",
        f"- Class imbalance ratio: {target_statistics['imbalance_ratio']}:1",
    ]
    for label, stats in target_statistics["distribution"].items():
        lines.append(f"  - {label}: {stats['count']} ({stats['percentage']}%)")
    return "\n".join(lines) + "\n"


def _render_numerical_summary(numerical_statistics: dict[str, Any]) -> str:
    """Render the "Numerical Feature Summary" Markdown table section."""
    if not numerical_statistics:
        return "## Numerical Feature Summary\n\nNo numerical features were analyzed.\n"

    lines = [
        "## Numerical Feature Summary\n",
        "| Feature | Mean | Median | Std | Min | Max | Skewness | Kurtosis |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for feature, stats in numerical_statistics.items():
        lines.append(
            f"| {feature} | {stats['mean']} | {stats['median']} | {stats['std']} | "
            f"{stats['min']} | {stats['max']} | {stats['skewness']} | {stats['kurtosis']} |"
        )
    return "\n".join(lines) + "\n"


def _render_categorical_summary(categorical_statistics: dict[str, Any]) -> str:
    """Render the "Categorical Feature Summary" Markdown table section."""
    if not categorical_statistics:
        return "## Categorical Feature Summary\n\nNo categorical features were analyzed.\n"

    lines = [
        "## Categorical Feature Summary\n",
        "| Feature | Cardinality | Mode |",
        "|---|---|---|",
    ]
    for feature, stats in categorical_statistics.items():
        lines.append(f"| {feature} | {stats['cardinality']} | {stats['mode']} |")
    return "\n".join(lines) + "\n"


def _render_correlation_summary(correlation_summary: dict[str, Any]) -> str:
    """Render the "Correlation Analysis" Markdown section."""
    lines = ["## Correlation Analysis\n"]

    lines.append("**Strongest positive correlations:**\n")
    if correlation_summary.get("strongest_positive"):
        lines.extend(
            f"- {pair['feature_a']} ↔ {pair['feature_b']}: {pair['correlation']}"
            for pair in correlation_summary["strongest_positive"]
        )
    else:
        lines.append("- None found.")

    lines.append("\n**Strongest negative correlations:**\n")
    if correlation_summary.get("strongest_negative"):
        lines.extend(
            f"- {pair['feature_a']} ↔ {pair['feature_b']}: {pair['correlation']}"
            for pair in correlation_summary["strongest_negative"]
        )
    else:
        lines.append("- None found.")

    return "\n".join(lines) + "\n"


def _render_key_observations(business_insights: dict[str, Any]) -> str:
    """Render a short "Key Observations" Markdown section summarizing insight counts."""
    categorical_count = len(business_insights.get("categorical_insights", []))
    numerical_count = len(business_insights.get("numerical_insights", []))
    return (
        "## Key Observations\n\n"
        f"- {categorical_count} categorical feature(s) showed a meaningful churn-rate gap.\n"
        f"- {numerical_count} numerical feature(s) showed a mean difference between "
        "churned and retained customers.\n"
    )


def _render_insights_markdown(business_insights: dict[str, Any]) -> str:
    """Render the standalone business insights and recommendations Markdown document.

    Args:
        business_insights: Output of ``generate_business_insights``.

    Returns:
        The complete Markdown document as a single string.
    """
    lines = [f"# Business Insights — {settings.project_name}\n"]

    lines.append("## Categorical Feature Insights\n")
    categorical_insights = business_insights.get("categorical_insights", [])
    if categorical_insights:
        lines.extend(f"- {insight['narrative']}" for insight in categorical_insights)
    else:
        lines.append("- No categorical features showed a meaningful churn-rate gap.")

    lines.append("\n## Numerical Feature Insights\n")
    numerical_insights = business_insights.get("numerical_insights", [])
    if numerical_insights:
        lines.extend(f"- {insight['narrative']}" for insight in numerical_insights)
    else:
        lines.append("- No numerical features showed a notable difference.")

    lines.append("\n## Recommendations\n")
    lines.extend(
        f"- {recommendation}" for recommendation in business_insights.get("recommendations", [])
    )

    return "\n".join(lines) + "\n"
