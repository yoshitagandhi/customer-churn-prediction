"""
===============================================================================
Customer Churn Prediction Platform
Overview Components
===============================================================================

Reusable executive overview components.

Responsibilities
----------------
• Executive dashboard
• KPI summaries
• Dataset summaries
• Model summaries
• Business insights

Presentation only.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

def render_overview_header(
    title: str,
    description: str | None = None,
) -> None:
    """
    Render overview header.
    """

    st.title(title)

    if description:

        st.caption(description)


def render_last_updated() -> None:
    """
    Display refresh timestamp.
    """

    st.caption(
        f"Last Updated: {datetime.now():%d %b %Y %H:%M:%S}"
    )
    
def render_kpi_overview(
    metrics: dict[str, Any],
) -> None:
    """
    Display key metrics.
    """

    columns = st.columns(
        len(metrics)
    )

    for column, (name, value) in zip(
        columns,
        metrics.items(),
    ):

        column.metric(
            name,
            value,
        )
        
def render_prediction_summary(
    total_predictions: int,
    churn_predictions: int,
) -> None:
    """
    Prediction summary.
    """

    st.subheader(
        "Prediction Summary"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Predictions",
        total_predictions,
    )

    col2.metric(
        "Churn Risk",
        churn_predictions,
    )
    
def render_dataset_summary(
    *,
    rows: int,
    columns: int,
    missing_values: float,
) -> None:
    """
    Dataset summary.
    """

    st.subheader(
        "Dataset"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{rows:,}",
    )

    c2.metric(
        "Columns",
        columns,
    )

    c3.metric(
        "Missing %",
        f"{missing_values:.2%}",
    )

def render_model_summary(
    *,
    model_name: str,
    accuracy: float,
) -> None:
    """
    Display model summary.
    """

    st.subheader(
        "Model"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Model",
        model_name,
    )

    col2.metric(
        "Accuracy",
        f"{accuracy:.2%}",
    )

def render_key_insights(
    insights: list[str],
) -> None:
    """
    Display business insights.
    """

    st.subheader(
        "Key Insights"
    )

    for insight in insights:

        st.markdown(
            f"- {insight}"
        )


def render_recommendations(
    recommendations: list[str],
) -> None:
    """
    Display recommendations.
    """

    st.subheader(
        "Recommendations"
    )

    for recommendation in recommendations:

        st.info(
            recommendation
        )
        
def render_overview(
    *,
    title: str,
    metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
    model_summary: dict[str, Any],
    insights: list[str],
    recommendations: list[str],
) -> None:
    """
    Render the complete overview dashboard.
    """

    render_overview_header(title)

    render_last_updated()

    render_kpi_overview(metrics)

    render_dataset_summary(**dataset_summary)

    render_model_summary(**model_summary)

    render_key_insights(insights)

    render_recommendations(recommendations)
    
    __all__ = [
    "render_overview",
    "render_overview_header",
    "render_last_updated",
    "render_kpi_overview",
    "render_prediction_summary",
    "render_dataset_summary",
    "render_model_summary",
    "render_key_insights",
    "render_recommendations",
]