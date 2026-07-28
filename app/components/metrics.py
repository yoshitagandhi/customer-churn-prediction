"""
===============================================================================
Customer Churn Prediction Platform
Metrics Components
===============================================================================

Purpose
-------
Reusable Streamlit metric components shared across the application.

These components provide a consistent presentation layer for displaying
business KPIs, model evaluation metrics, dataset statistics, prediction
summaries, and operational health indicators.

Architecture
------------
Presentation Layer Only

No business logic or machine learning logic should exist in this module.

Responsibilities
----------------
• Render individual metrics
• Render responsive metric grids
• Format metric values
• Display delta metrics
• Display percentage metrics
• Display business KPIs
• Validate metric definitions

Used By
-------
• dashboard.py
• evaluation.py
• prediction.py
• batch_prediction.py
===============================================================================
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_COLUMNS = 4

SUPPORTED_FORMATS = {
    ".2%",
    ".1%",
    ".3f",
    ".2f",
    ".1f",
    ",",
    None,
}

def validate_metric(
    metric: dict[str, Any],
) -> None:
    """
    Validate a metric definition.

    Parameters
    ----------
    metric
        Metric configuration dictionary.

    Raises
    ------
    ValueError
        If required keys are missing.
    """

    required_keys = {
        "label",
        "value",
    }

    missing = required_keys.difference(metric)

    if missing:

        raise ValueError(
            "Metric is missing required keys: "
            f"{', '.join(sorted(missing))}"
        )


def format_metric_value(
    value: Any,
    value_format: str | None = None,
) -> str:
    """
    Format a metric value for display.

    Parameters
    ----------
    value
        Metric value.

    value_format
        Optional formatting string.

    Returns
    -------
    str
    """

    if value is None:

        return "N/A"

    if value_format is None:

        return str(value)

    if value_format not in SUPPORTED_FORMATS:

        return str(value)

    try:

        return format(
            value,
            value_format,
        )

    except Exception:

        return str(value)

def render_metric(
    *,
    label: str,
    value: Any,
    value_format: str | None = None,
    delta: str | float | None = None,
    delta_color: str = "normal",
    help_text: str | None = None,
) -> None:
    """
    Render a single Streamlit metric.

    Parameters
    ----------
    label
        Metric label.

    value
        Metric value.

    value_format
        Optional formatting string.

    delta
        Optional delta value.

    delta_color
        Streamlit delta color mode.

    help_text
        Optional tooltip.
    """

    formatted_value = format_metric_value(
        value=value,
        value_format=value_format,
    )

    st.metric(
        label=label,
        value=formatted_value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )

def render_metrics_grid(
    metrics: list[dict[str, Any]],
    columns: int = DEFAULT_COLUMNS,
) -> None:
    """
    Render a responsive grid of metrics.

    Parameters
    ----------
    metrics
        List of metric definitions.

    columns
        Number of metrics per row.
    """

    if not metrics:

        st.info(
            "No metrics available."
        )

        return

    for row_start in range(
        0,
        len(metrics),
        columns,
    ):

        row_metrics = metrics[
            row_start : row_start + columns
        ]

        row = st.columns(
            len(row_metrics)
        )

        for column, metric in zip(
            row,
            row_metrics,
            strict=True,
        ):

            validate_metric(
                metric,
            )

            with column:

                render_metric(
                    label=metric["label"],
                    value=metric["value"],
                    value_format=metric.get(
                        "format",
                    ),
                    delta=metric.get(
                        "delta",
                    ),
                    delta_color=metric.get(
                        "delta_color",
                        "normal",
                    ),
                    help_text=metric.get(
                        "help",
                    ),
                )

def render_percentage_metric(
    label: str,
    value: float,
    *,
    delta: str | None = None,
    help_text: str | None = None,
) -> None:
    """
    Render a percentage metric.
    """

    render_metric(
        label=label,
        value=value,
        value_format=".2%",
        delta=delta,
        help_text=help_text,
    )

def render_delta_metric(
    label: str,
    value: Any,
    delta: str | float,
    *,
    value_format: str | None = None,
    delta_color: str = "normal",
) -> None:
    """
    Render a metric with delta.
    """

    render_metric(
        label=label,
        value=value,
        value_format=value_format,
        delta=delta,
        delta_color=delta_color,
    )

def render_progress_metric(
    label: str,
    value: float,
    *,
    value_format: str = ".2%",
) -> None:
    """
    Render a metric accompanied by
    a progress bar.
    """

    render_metric(
        label=label,
        value=value,
        value_format=value_format,
    )

    progress = max(
        0.0,
        min(
            float(value),
            1.0,
        ),
    )

    st.progress(
        progress,
    )

def render_business_metric(
    *,
    label: str,
    value: Any,
    description: str,
    value_format: str | None = None,
) -> None:
    """
    Render a business KPI with description.
    """

    render_metric(
        label=label,
        value=value,
        value_format=value_format,
    )

    st.caption(
        description,
    )

def render_metric_group(
    title: str,
    metrics: list[dict[str, Any]],
    *,
    columns: int = DEFAULT_COLUMNS,
) -> None:
    """
    Render a titled group of metrics.

    Parameters
    ----------
    title
        Section heading.

    metrics
        Collection of metric definitions.

    columns
        Number of metric columns.
    """

    st.subheader(
        title,
    )

    render_metrics_grid(
        metrics=metrics,
        columns=columns,
    )

def render_model_metrics(
    *,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
) -> None:
    """
    Render standard model evaluation metrics.
    """

    render_metric_group(
        title="Model Performance",
        metrics=[
            {
                "label": "Accuracy",
                "value": accuracy,
                "format": ".2%",
            },
            {
                "label": "Precision",
                "value": precision,
                "format": ".2%",
            },
            {
                "label": "Recall",
                "value": recall,
                "format": ".2%",
            },
            {
                "label": "F1 Score",
                "value": f1_score,
                "format": ".2%",
            },
        ],
    )


def render_dataset_metrics(
    *,
    samples: int,
    features: int,
    missing_values: int,
) -> None:
    """
    Render dataset statistics.
    """

    render_metric_group(
        title="Dataset Summary",
        metrics=[
            {
                "label": "Samples",
                "value": samples,
                "format": ",",
            },
            {
                "label": "Features",
                "value": features,
                "format": ",",
            },
            {
                "label": "Missing Values",
                "value": missing_values,
                "format": ",",
            },
        ],
        columns=3,
    )

def render_prediction_metrics(
    *,
    probability: float,
    prediction: str,
    confidence: float,
) -> None:
    """
    Render prediction results.
    """

    render_metric_group(
        title="Prediction Summary",
        metrics=[
            {
                "label": "Prediction",
                "value": prediction,
            },
            {
                "label": "Probability",
                "value": probability,
                "format": ".2%",
            },
            {
                "label": "Confidence",
                "value": confidence,
                "format": ".2%",
            },
        ],
        columns=3,
    )


def render_business_metrics(
    metrics: list[dict[str, Any]],
) -> None:
    """
    Render business KPI metrics.
    """

    render_metric_group(
        title="Business KPIs",
        metrics=metrics,
    )

def render_empty_metrics() -> None:
    """
    Display an empty metrics state.
    """

    st.info(
        "No metrics are available."
    )

__all__ = [
    "render_metric",
    "render_metrics_grid",
    "render_metric_group",
    "render_percentage_metric",
    "render_delta_metric",
    "render_progress_metric",
    "render_business_metric",
    "render_model_metrics",
    "render_dataset_metrics",
    "render_prediction_metrics",
    "render_business_metrics",
    "render_empty_metrics",
]