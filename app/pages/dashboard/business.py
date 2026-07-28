"""
Customer Churn Prediction Platform
Dashboard Business Insights

Business-oriented dashboard components.

Responsibilities
----------------
• Customer portfolio summary
• Model status
• Dataset quality
• Platform health
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
import streamlit as st

from app.components.cards import render_business_insight_card
from app.components.metrics import render_metrics_grid

from .models import DashboardData

@dataclass(frozen=True, slots=True)
class ModelStatus:
    """
    Production model status configuration.
    """

    minimum_score: float
    title: str
    narrative: str


MODEL_STATUSES: Final = (
    ModelStatus(
        0.62,
        "Strong Signal",
        "The production model is delivering a strong, consistent retention signal.",
    ),
    ModelStatus(
        0.55,
        "Operational",
        "The production model is active and ready to support retention prioritization.",
    ),
    ModelStatus(
        0.00,
        "Live Signal",
        "The production model is active and its customer-risk signals are available.",
    ),
)

def render_customer_portfolio(
    dashboard: DashboardData,
) -> None:
    """
    Display customer churn portfolio.
    """

    st.subheader("Customer Portfolio")

    evaluation = dashboard.evaluation
    predictions = (
        evaluation.get("predictions")
        if isinstance(evaluation, dict)
        else getattr(evaluation, "predictions", None)
    )

    if not isinstance(predictions, pd.DataFrame) or predictions.empty or "prediction" not in predictions:

        st.info(
            "Prediction summary is unavailable."
        )

        return

    total = len(predictions)

    high_risk = int(pd.to_numeric(predictions["prediction"], errors="coerce").fillna(0).eq(1).sum())

    low_risk = total - high_risk

    high_risk_rate = high_risk / total if total else 0.0

    render_business_insight_card(
        {
            "title": "Customer Risk",
            "narrative": (
                f"{high_risk_rate:.1%} of customers "
                "are predicted to churn."
            ),
            "recommendations": [
                "Prioritize retention campaigns.",
                "Review high-value customers.",
                "Monitor churn weekly.",
            ],
        }
    )

    left, right = st.columns(2)

    with left:

        st.metric(
            "High Risk",
            f"{high_risk:,}",
        )

    with right:

        st.metric(
            "Low Risk",
            f"{low_risk:,}",
        )

def render_model_status(
    dashboard: DashboardData,
) -> None:
    """
    Display production model status.
    """

    score = dashboard.evaluation["metrics"]["f1"]

    status = next(
        (
            item
            for item in MODEL_STATUSES
            if score >= item.minimum_score
        ),
        MODEL_STATUSES[-1],
    )

    theme = {
        "Strong Signal": "healthy",
        "Operational": "stable",
        "Live Signal": "stable",
    }[status.title]
    st.markdown(
        f"""
        <div class="attention-heading"><i>✦</i> Model Status</div>
        <section class="status-spotlight {theme}">
            <div class="spotlight-kicker">PRODUCTION MODEL</div>
            <div class="spotlight-title">{status.title}</div>
            <p class="spotlight-copy">{status.narrative}</p>
            <div class="spotlight-score">F1 {score:.0%}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_dataset_quality(
    dashboard: DashboardData,
) -> None:
    """
    Display dataset quality metrics.
    """

    st.subheader("Dataset Quality")

    features = dashboard.features

    total_records = len(features)

    total_features = features.shape[1]

    missing_values = int(
        features.isna().sum().sum()
    )

    completeness = (
        1
        - (
            missing_values
            / (
                total_records
                * total_features
            )
        )
    )

    metrics = [
        {
            "label": "Records",
            "value": total_records,
            "format": ",",
        },
        {
            "label": "Features",
            "value": total_features,
            "format": ",",
        },
        {
            "label": "Missing Values",
            "value": missing_values,
            "format": ",",
        },
        {
            "label": "Completeness",
            "value": completeness,
            "format": ".2%",
        },
    ]

    render_metrics_grid(metrics)

def render_platform_health(
    dashboard: DashboardData,
) -> None:
    """
    Display overall platform readiness.
    """

    metadata = dashboard.metadata

    checks = (
        (
            "Production model available",
            metadata is not None,
        ),
        (
            "Training metadata available",
            bool(metadata),
        ),
        (
            "Model version available",
            bool(
                metadata.get(
                    "model_version"
                )
            ),
        ),
        (
            "Training completed",
            bool(
                metadata.get(
                    "training_date"
                )
            ),
        ),
    )

    check_cards = "".join(
        '<div class="health-check{}"><b>{}</b>{}</div>'.format(
            "" if passed else " offline",
            "● ONLINE" if passed else "● ACTION NEEDED",
            label,
        )
        for label, passed in checks
        if passed
    )
    st.markdown(
        f"""
        <div class="attention-heading"><i>✓</i> Platform Health</div>
        <div class="health-grid">{check_cards}</div>
        """,
        unsafe_allow_html=True,
    )

def render_business_dashboard(
    dashboard: DashboardData,
) -> None:
    """
    Render all business insight sections.
    """

    sections = (
        lambda: render_customer_portfolio(
            dashboard,
        ),
        lambda: render_model_status(
            dashboard,
        ),
        lambda: render_dataset_quality(
            dashboard,
        ),
        lambda: render_platform_health(
            dashboard,
        ),
    )

    for index, section in enumerate(sections):

        section()

        if index != len(sections) - 1:

            st.divider()

__all__ = [
    "render_business_dashboard",
]
