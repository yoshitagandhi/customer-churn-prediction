"""
Customer Churn Prediction Platform
Dashboard Overview

Executive overview components displayed on the landing page.

Responsibilities
----------------
• Platform overview
• Production model summary
• Dataset summary
• Deployment readiness
"""

from __future__ import annotations

import streamlit as st

from app.components.cards import (
    render_business_insight_card,
)

from .models import DashboardData

def render_platform_summary() -> None:
    """
    Display a high-level overview of the platform.
    """

    st.subheader("Platform Summary")

    st.write(
        (
            "The Customer Churn Prediction Platform provides "
            "machine learning powered customer churn prediction, "
            "batch inference, explainability, model evaluation, "
            "and business intelligence through an integrated "
            "production-ready dashboard."
        )
    )

def render_production_model(
    dashboard: DashboardData,
) -> None:
    """
    Display deployed model information.
    """

    metadata = dashboard.metadata

    st.subheader("Production Model")

    left, right = st.columns(2)

    with left:

        st.write(
            f"**Model Name:** "
            f"{metadata.get('model_name', 'Unknown')}"
        )

        st.write(
            f"**Version:** "
            f"{metadata.get('model_version', 'N/A')}"
        )

        st.write(
            f"**Training Date:** "
            f"{metadata.get('training_date', 'N/A')}"
        )

    with right:

        st.write(
            f"**Features:** "
            f"{metadata.get('feature_count', 'N/A')}"
        )

        training_samples = metadata.get(
            "training_samples"
        )

        validation_samples = metadata.get(
            "validation_samples"
        )

        st.write(
            f"**Training Samples:** "
            f"{training_samples:,}"
            if training_samples
            else "**Training Samples:** N/A"
        )

        st.write(
            f"**Validation Samples:** "
            f"{validation_samples:,}"
            if validation_samples
            else "**Validation Samples:** N/A"
        )

def render_dataset_summary(
    dashboard: DashboardData,
) -> None:
    """
    Display validation dataset statistics.
    """

    target = dashboard.target

    st.subheader("Validation Dataset")

    from configs.config import settings

    from pandas.api.types import is_numeric_dtype
    
    target = dashboard.target

    total_customers = len(target)

    if is_numeric_dtype(target):
        churn_customers = int(target.sum())
    else:
        churn_customers = int(
            (target.astype(str) == settings.positive_label).sum()
            )

    retained_customers = total_customers - churn_customers

    churn_rate = (
        churn_customers
        / total_customers
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Customers",
            f"{total_customers:,}",
        )

    with c2:

        st.metric(
            "Churn",
            f"{churn_customers:,}",
        )

    with c3:

        st.metric(
            "Retained",
            f"{retained_customers:,}",
        )

    with c4:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2%}",
        )

def render_executive_overview(
    dashboard: DashboardData,
) -> None:
    """
    Render the executive overview section.
    """

    sections = (
        render_platform_summary,
        lambda: render_production_model(
            dashboard
        ),
        lambda: render_dataset_summary(
            dashboard
        ),
    )

    for index, section in enumerate(sections):

        section()

        if index != len(sections) - 1:
            st.divider()

__all__ = [
    "render_executive_overview",
]
