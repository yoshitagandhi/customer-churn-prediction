"""
Customer Churn Prediction Platform
Dashboard Analytics

Analytics visualizations for the Executive Dashboard.

Responsibilities
----------------
• Dataset analytics
• Feature analytics
• Target analytics
• Metadata analytics

This module contains rendering only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...components.tables import render_dataframe_table

from app.pages.dashboard.formatters import (
    format_number,
    format_percentage,
)

from app.pages.dashboard.helpers import (
    calculate_dataset_summary,
)

from app.pages.dashboard.models import DashboardData

def render_dataset_overview(
    dashboard: DashboardData,
) -> None:
    """
    Display dataset statistics.
    """

    st.subheader("Dataset Analytics")

    summary = calculate_dataset_summary(
        dashboard.features,
        dashboard.target,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Records",
            format_number(summary["records"]),
        )

    with c2:
        st.metric(
            "Features",
            format_number(summary["features"]),
        )

    with c3:
        st.metric(
            "Completeness",
            format_percentage(
                summary["completeness"],
            ),
        )

def render_feature_preview(
    dashboard: DashboardData,
) -> None:
    """
    Display a preview of validation features.
    """

    st.subheader("Validation Features")

    render_dataframe_table(
        dashboard.features.head(10),
    )

def render_target_distribution(
    dashboard: DashboardData,
) -> None:
    """
    Display target distribution.
    """

    st.subheader("Target Distribution")

    distribution = (
        dashboard.target
        .value_counts()
        .rename_axis("Target")
        .reset_index(name="Count")
    )

    render_dataframe_table(distribution)

def render_training_metadata(
    dashboard: DashboardData,
) -> None:
    """
    Display cached training metadata.
    """

    st.subheader("Training Metadata")

    metadata = dashboard.metadata

    if not metadata:

        st.info(
            "Training metadata unavailable."
        )

        return

    metadata_frame = pd.DataFrame(
        [{"Field": key, "Value": value} for key, value in metadata.items()]
    )
    render_dataframe_table(metadata_frame)

def render_analytics_dashboard(
    dashboard: DashboardData,
) -> None:
    """
    Render analytics section.
    """

    sections = (
        lambda: render_dataset_overview(
            dashboard,
        ),
        lambda: render_feature_preview(
            dashboard,
        ),
        lambda: render_target_distribution(
            dashboard,
        ),
        lambda: render_training_metadata(
            dashboard,
        ),
    )

    for index, section in enumerate(sections):

        section()

        if index != len(sections) - 1:
            st.divider()

__all__ = [
    "render_analytics_dashboard",
]