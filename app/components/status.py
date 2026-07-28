"""
===============================================================================
Customer Churn Prediction Platform
Status Components
===============================================================================

Reusable status indicators shared throughout the application.

Responsibilities
----------------
• Display persistent application status
• Display model readiness
• Display dataset health
• Display processing status
• Display infrastructure health

Presentation only.
===============================================================================
"""

from __future__ import annotations

from enum import Enum

import streamlit as st

class StatusLevel(str, Enum):
    """
    Application status levels.
    """

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


STATUS_ICONS = {
    StatusLevel.SUCCESS: "",
    StatusLevel.WARNING: "",
    StatusLevel.ERROR: "",
    StatusLevel.INFO: "",
}

def render_status(
    *,
    title: str,
    message: str,
    level: StatusLevel,
) -> None:
    """
    Render a standardized status message.
    """

    icon = STATUS_ICONS[level]

    st.markdown(
        f"### {icon} {title}"
    )

    if level is StatusLevel.SUCCESS:

        st.success(message)

    elif level is StatusLevel.WARNING:

        st.warning(message)

    elif level is StatusLevel.ERROR:

        st.error(message)

    else:

        st.info(message)


def render_status_badge(
    label: str,
    level: StatusLevel,
) -> None:
    """
    Render compact status badge.
    """

    st.caption(
        f"{STATUS_ICONS[level]} {label}"
    )

def render_model_status(
    loaded: bool,
) -> None:
    """
    Display model availability.
    """

    if loaded:

        render_status(
            title="Model Ready",
            message="Prediction model loaded.",
            level=StatusLevel.SUCCESS,
        )

    else:

        render_status(
            title="Model Missing",
            message="Load a trained model.",
            level=StatusLevel.ERROR,
        )


def render_model_version(
    version: str,
) -> None:
    """
    Display model version.
    """

    st.metric(
        "Model Version",
        version,
    )

def render_dataset_status(
    rows: int,
    columns: int,
) -> None:
    """
    Dataset summary.
    """

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        f"{rows:,}",
    )

    col2.metric(
        "Columns",
        columns,
    )


def render_missing_values(
    missing_percentage: float,
) -> None:
    """
    Missing values indicator.
    """

    if missing_percentage < 0.05:

        render_status_badge(
            "Healthy",
            StatusLevel.SUCCESS,
        )

    elif missing_percentage < 0.20:

        render_status_badge(
            "Warning",
            StatusLevel.WARNING,
        )

    else:

        render_status_badge(
            "Poor Quality",
            StatusLevel.ERROR,
        )

def render_cache_status(
    cached: bool,
) -> None:
    """
    Cache indicator.
    """

    render_status_badge(
        "Cache Enabled" if cached else "Cache Disabled",
        StatusLevel.SUCCESS if cached else StatusLevel.WARNING,
    )


def render_application_status() -> None:
    """
    Overall application health.
    """

    render_status(
        title="Application",
        message="Running normally.",
        level=StatusLevel.SUCCESS,
    )
    
    __all__ = [
    "StatusLevel",
    "render_status",
    "render_status_badge",
    "render_model_status",
    "render_model_version",
    "render_dataset_status",
    "render_missing_values",
    "render_cache_status",
    "render_application_status",
]