"""SHAP explainability report generation.

Turns global/local SHAP explanations and their business
interpretations into three artifacts: ``shap_report.md``,
``feature_importance.csv``, and ``customer_explanations.json``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from app.services.prediction_service import PredictionResult

logger = get_logger(__name__)


def generate_shap_report(
    ranked_features: list[dict[str, Any]],
    example_explanations: list[dict[str, Any]],
    figure_paths: dict[str, Any],
    report_path: Path = settings.shap_report_path,
    feature_importance_path: Path = settings.feature_importance_csv_path,
    customer_explanations_path: Path = settings.customer_explanations_path,
) -> dict[str, Path]:
    """Generate the Markdown, CSV, and JSON explainability artifacts.

    Args:
        ranked_features: Output of
            :func:`src.explainability.feature_importance.rank_features`.
        example_explanations: A list of combined prediction +
            business-insight dictionaries (one per example customer),
            each with keys "customer_index", "prediction", and
            "business_insights".
        figure_paths: Mapping of figure identifiers to their saved
            file paths, for reference in the report.
        report_path: Destination for the Markdown report. Defaults to
            ``settings.shap_report_path``.
        feature_importance_path: Destination for the feature
            importance CSV. Defaults to
            ``settings.feature_importance_csv_path``.
        customer_explanations_path: Destination for the customer
            explanations JSON. Defaults to
            ``settings.customer_explanations_path``.

    Returns:
        A mapping with keys "report_markdown", "feature_importance_csv",
        and "customer_explanations_json" pointing to the generated
        file paths.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    customer_explanations_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(ranked_features).to_csv(feature_importance_path, index=False)
    customer_explanations_path.write_text(
        json.dumps(example_explanations, indent=2, default=str), encoding="utf-8"
    )
    report_path.write_text(
        _render_markdown(ranked_features, example_explanations, figure_paths), encoding="utf-8"
    )

    logger.info(
        "Reports generated: %s, %s, %s.",
        report_path,
        feature_importance_path,
        customer_explanations_path,
    )
    return {
        "report_markdown": report_path,
        "feature_importance_csv": feature_importance_path,
        "customer_explanations_json": customer_explanations_path,
    }


def _render_markdown(
    ranked_features: list[dict[str, Any]],
    example_explanations: list[dict[str, Any]],
    figure_paths: dict[str, Any],
) -> str:
    """Render the full SHAP report as a Markdown document.

    Args:
        ranked_features: Global feature ranking.
        example_explanations: Example customer explanations.
        figure_paths: Mapping of figure identifiers to saved paths.

    Returns:
        The complete Markdown document as a single string.
    """
    sections = [
        f"# SHAP Explainability Report — {settings.project_name}",
        f"\nGenerated at: {datetime.now(UTC).isoformat()}\n",
        _render_global_summary_section(ranked_features),
        _render_example_customers_section(example_explanations),
        _render_figures_section(figure_paths),
    ]
    return "\n".join(sections)


def _render_global_summary_section(ranked_features: list[dict[str, Any]]) -> str:
    """Render the "Global Explanation Summary" and "Top Churn-Driving Features" sections."""
    lines = [
        "## Global Explanation Summary\n",
        "The table below ranks features by their average impact (mean absolute SHAP "
        "value) on the model's churn predictions across the evaluated customers.\n",
        "## Top Churn-Driving Features\n",
        "| Rank | Feature | Mean |SHAP| |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item['rank']} | {item['feature']} | {item['mean_abs_shap']} |"
        for item in ranked_features
    )
    return "\n".join(lines) + "\n"


def _render_example_customers_section(example_explanations: list[dict[str, Any]]) -> str:
    """Render the "Example Customer Explanations" and "Business Recommendations" sections."""
    lines = ["## Example Customer Explanations\n"]
    for example in example_explanations:
        prediction = example["prediction"]
        insights = example["business_insights"]
        lines.append(f"### Customer {example['customer_index']}\n")
        lines.append(f"- Predicted churn probability: prediction: PredictionResult")
        lines.append(f"- Risk level: {insights['risk_level']}")
        lines.append(f"- {insights['narrative']}")
        lines.append("- Suggested business actions:")
        lines.extend(f"  - {recommendation}" for recommendation in insights["recommendations"])
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_figures_section(figure_paths: dict[str, Any]) -> str:
    """Render the "Figures Generated" Markdown section."""
    lines = ["## Figures Generated\n"]
    lines.extend(f"- **{name}**: `{path}`" for name, path in figure_paths.items())
    return "\n".join(lines) + "\n"
