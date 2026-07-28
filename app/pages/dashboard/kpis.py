"""
Customer Churn Prediction Platform
Dashboard KPIs

Render model evaluation metrics and production health.

Responsibilities
----------------
• Executive KPI snapshot
• Model performance metrics
• Advanced evaluation metrics
• Production model health
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import streamlit as st

from app.components.cards import render_business_insight_card
from app.components.metrics import render_metrics_grid

from .models import DashboardData

@dataclass(frozen=True, slots=True)
class HealthStatus:
    """
    Model health configuration.
    """

    minimum_score: float
    title: str
    description: str


HEALTH_LEVELS: Final = (
    HealthStatus(
        0.62,
        "Strong",
        "The model shows a useful balance of churn detection and overall classification performance for this decision-support workflow.",
    ),
    HealthStatus(
        0.50,
        "Operational",
        "The model is operational, with clear opportunities for threshold and feature refinement.",
    ),
    HealthStatus(
        0.00,
        "Active",
        "The model is live and contributing a useful retention signal.",
    ),
)

def render_kpi_snapshot(
    dashboard: DashboardData,
) -> None:
    """
    Display the primary executive KPIs.
    """

    evaluation = dashboard.evaluation["metrics"]

    st.subheader("Executive KPI Snapshot")

    left, right = st.columns(2)

    with left:

        st.metric(
            "Accuracy",
            f"{evaluation['accuracy']:.2%}",
        )

        st.metric(
            "F1 Score",
            f"{evaluation['f1']:.2%}",
        )

    with right:

        st.metric(
            "ROC-AUC",
            f"{evaluation["roc_auc"]:.3f}",
        )

        st.metric(
            "Recall",
            f"{evaluation["recall"]:.2%}",
        )

def render_model_performance(
    dashboard: DashboardData,
) -> None:
    """
    Display the primary performance metrics.
    """

    evaluation = dashboard.evaluation["metrics"]

    st.subheader("Model Performance")

    metrics = [
        {
            "label": "Accuracy",
            "value": evaluation["accuracy"],
            "format": ".2%",
        },
        {
            "label": "Precision",
            "value": evaluation["precision"],
            "format": ".2%",
        },
        {
            "label": "Recall",
            "value": evaluation["recall"],
            "format": ".2%",
        },
        {
            "label": "F1 Score",
            "value": evaluation["f1"],
            "format": ".2%",
        },
    ]

    render_metrics_grid(metrics)

def render_advanced_metrics(
    dashboard: DashboardData,
) -> None:
    """
    Display advanced evaluation metrics.
    """

    evaluation = dashboard.evaluation["metrics"]

    st.subheader("Advanced Metrics")

    metrics = [
        {
            "label": "ROC-AUC",
            "value": evaluation["roc_auc"],
            "format": ".3f",
        },
        {
            "label": "PR-AUC",
            "value": evaluation["pr_auc"],
            "format": ".3f",
        },
        {
            "label": "Specificity",
            "value": evaluation["specificity"],
            "format": ".2%",
        },
        {
            "label": "Balanced Accuracy",
            "value": evaluation["balanced_accuracy"],
            "format": ".2%",
        },
    ]

    render_metrics_grid(metrics)

def render_model_health(
    dashboard: DashboardData,
) -> None:
    """
    Display production model health.
    """

    score = dashboard.evaluation["metrics"]["f1"]

    health = next(
        (
            level
            for level in HEALTH_LEVELS
            if score >= level.minimum_score
        ),
        HEALTH_LEVELS[-1],
    )

    theme = "healthy" if health.title == "Strong" else "stable"
    st.markdown(
        f"""
        <div class="attention-heading"><i>♥</i> Model Health</div>
        <section class="status-spotlight {theme}">
            <div class="spotlight-kicker">PERFORMANCE SNAPSHOT</div>
            <div class="spotlight-title">{health.title}</div>
            <p class="spotlight-copy">{health.description}</p>
            <div class="spotlight-score">F1 {score:.0%}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_kpi_dashboard(
    dashboard: DashboardData,
) -> None:
    """
    Render the complete KPI dashboard.
    """

    sections = (
        lambda: render_kpi_snapshot(
            dashboard,
        ),
        lambda: render_model_performance(
            dashboard,
        ),
        lambda: render_advanced_metrics(
            dashboard,
        ),
        lambda: render_model_health(
            dashboard,
        ),
    )

    for index, section in enumerate(sections):

        section()

        if index != len(sections) - 1:
            st.divider()

__all__ = [
    "render_kpi_dashboard",
]
