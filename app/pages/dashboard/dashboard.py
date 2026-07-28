"""
Customer Churn Prediction Platform
Executive Dashboard

Main entry point for the Executive Dashboard.

Responsibilities
----------------
• Configure the dashboard page
• Load dashboard resources
• Orchestrate dashboard sections
• Handle top-level errors

No business logic should live in this module.
"""

from __future__ import annotations

import streamlit as st

from configs.logging_config import get_logger

from .business import render_business_dashboard
from .kpis import render_kpi_dashboard
from .loaders import load_dashboard_data
from .navigation import render_navigation_hub
from .overview import render_executive_overview
from .analytics import render_analytics_dashboard

logger = get_logger(__name__)

PAGE_TITLE = "Retention Intelligence"

PAGE_DESCRIPTION = (
    "A unified overview of model performance, business insights, "
    "analytics, and production health."
)

def render_header() -> None:
    """
    Render the dashboard header.
    """

    st.markdown(
        f"""
        <section class="page-hero">
            <div class="eyebrow">CUSTOMER RETENTION / MODEL OPERATIONS</div>
            <h1>{PAGE_TITLE}</h1>
            <p>{PAGE_DESCRIPTION}</p>
            <div class="hero-status"><span></span> Production model connected · signals are fresh</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_sections(dashboard) -> None:
    """
    Render every dashboard section.
    """

    sections = (
        lambda: render_executive_overview(dashboard),
        lambda: render_kpi_dashboard(dashboard),
        lambda: render_business_dashboard(dashboard),
        lambda: render_analytics_dashboard(dashboard),
        render_navigation_hub,
    )

    for index, section in enumerate(sections):

        section()

        if index != len(sections) - 1:
            st.divider()

def render_dashboard_page() -> None:
    """
    Render the Executive Dashboard.
    """

    render_header()

    try:
        dashboard = load_dashboard_data()
        render_sections(dashboard)

    except Exception as exc:
        logger.exception("Failed to render Executive Dashboard.")
        st.error("Dashboard data could not be loaded. Check the application logs for details.")
        st.caption(f"Error reference: {type(exc).__name__}")

__all__ = [
    "render_dashboard_page",
]
