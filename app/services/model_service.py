"""Export service.

Converts prediction results into downloadable formats. No UI code —
returns bytes/strings for Streamlit's download widgets to serve.
Designed so new export formats can be added without touching existing
ones.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


def export_predictions_csv(dataframe: pd.DataFrame) -> bytes:
    """Export a batch prediction DataFrame as CSV bytes.

    Args:
        dataframe: The predictions (and recommendations) DataFrame.

    Returns:
        UTF-8 encoded CSV content, ready for a download button.
    """
    logger.info("Export completed.")
    return dataframe.to_csv(index=False).encode("utf-8")


def export_prediction_json(prediction_result: dict[str, Any]) -> bytes:
    """Export a single prediction result as JSON bytes.

    Args:
        prediction_result: A structured prediction/recommendation
            result dictionary.

    Returns:
        UTF-8 encoded JSON content, ready for a download button.
    """
    payload = {"generated_at": datetime.now(UTC).isoformat(), **prediction_result}
    logger.info("Export completed.")
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def export_prediction_markdown_summary(
    prediction_result: dict[str, Any], business_insights: dict[str, Any] | None = None
) -> bytes:
    """Export a single prediction result as a human-readable Markdown summary.

    Args:
        prediction_result: A structured prediction/recommendation
            result dictionary.
        business_insights: Optional business-friendly explanation
            (see
            :func:`app.services.explanation_service.explain_customer_prediction`).

    Returns:
        UTF-8 encoded Markdown content, ready for a download button.
    """
    lines = [
        f"# Prediction Summary — {settings.project_name}",
        f"\nGenerated at: {datetime.now(UTC).isoformat()}\n",
        f"- Predicted probability: {prediction_result.get('predicted_probability')}",
        f"- Predicted class: {prediction_result.get('predicted_class')}",
        f"- Risk level: {prediction_result.get('risk_level')}",
        f"- Recommended action: {prediction_result.get('recommended_action')}",
    ]
    if business_insights:
        lines.append(f"\n## Explanation\n\n{business_insights.get('narrative', '')}")
        if business_insights.get("recommendations"):
            lines.append("\n### Suggested Business Actions\n")
            lines.extend(f"- {item}" for item in business_insights["recommendations"])

    logger.info("Export completed.")
    return "\n".join(lines).encode("utf-8")
