"""
===============================================================================
Customer Churn Prediction Platform
Alert Components
===============================================================================

Reusable alert and notification components shared throughout the
Streamlit application.

Responsibilities
----------------
• Display informational messages
• Display success, warning, and error notifications
• Present validation feedback
• Handle exceptions consistently

This module contains presentation logic only.
===============================================================================
"""

from __future__ import annotations

import traceback

import streamlit as st

DEFAULT_ICON = "ℹ️"

ALERT_ICONS = {
    "info": "ℹ️",
    "success": "",
    "warning": "️",
    "error": "",
}

def _format_message(
    title: str,
    message: str,
) -> str:
    """
    Build a standardized alert message.
    """

    return f"**{title}**\n\n{message}"

def _render_alert(
    *,
    alert_type: str,
    title: str,
    message: str,
) -> None:
    """
    Render a standardized Streamlit alert.
    """

    formatted = _format_message(
        title,
        message,
    )

    match alert_type:

        case "success":
            st.success(formatted)

        case "warning":
            st.warning(formatted)

        case "error":
            st.error(formatted)

        case _:
            st.info(formatted)
            
def render_info_alert(
    *,
    title: str,
    message: str,
) -> None:
    """
    Display an informational alert.
    """

    _render_alert(
        alert_type="info",
        title=title,
        message=message,
    )

def render_success_alert(
    *,
    title: str,
    message: str,
) -> None:
    """
    Display a success alert.
    """

    _render_alert(
        alert_type="success",
        title=title,
        message=message,
    )

def render_warning_alert(
    *,
    title: str,
    message: str,
) -> None:
    """
    Display a warning alert.
    """

    _render_alert(
        alert_type="warning",
        title=title,
        message=message,
    )

def render_error_alert(
    *,
    title: str,
    message: str,
) -> None:
    """
    Display an error alert.
    """

    _render_alert(
        alert_type="error",
        title=title,
        message=message,
    )

def render_exception_alert(
    exception: Exception,
    *,
    title: str = "Unexpected Error",
    show_traceback: bool = False,
) -> None:
    """
    Display a formatted exception.
    """

    render_error_alert(
        title=title,
        message=str(exception),
    )

    if show_traceback:

        with st.expander(
            "Technical Details",
            expanded=False,
        ):

            st.code(
                "".join(
                    traceback.format_exception(
                        type(exception),
                        exception,
                        exception.__traceback__,
                    )
                ),
                language="text",
            )

def render_validation_alert(
    validation_errors: list[str],
) -> None:
    """
    Display validation failures.
    """

    if not validation_errors:
        return

    st.warning(
        "Validation failed."
    )

    for error in validation_errors:

        st.markdown(
            f"- {error}"
        )

def render_prediction_alert(
    *,
    prediction: str,
    probability: float,
) -> None:
    """
    Display prediction outcome.
    """

    if prediction.lower() == "churn":

        render_warning_alert(
            title="High Churn Risk",
            message=(
                f"The predicted probability of churn is "
                f"{probability:.2%}."
            ),
        )

    else:

        render_success_alert(
            title="Low Churn Risk",
            message=(
                f"The predicted probability of churn is "
                f"{probability:.2%}."
            ),
        )

def render_model_status_alert(
    *,
    is_loaded: bool,
    model_name: str,
) -> None:
    """
    Display model availability.
    """

    if is_loaded:

        render_success_alert(
            title="Model Ready",
            message=f"{model_name} loaded successfully.",
        )

    else:

        render_error_alert(
            title="Model Unavailable",
            message=f"{model_name} could not be loaded.",
        )

def render_data_quality_alert(
    *,
    missing_percentage: float,
) -> None:
    """
    Warn when missing values exceed thresholds.
    """

    if missing_percentage < 0.05:

        render_success_alert(
            title="Data Quality",
            message="No significant missing values detected.",
        )

    elif missing_percentage < 0.20:

        render_warning_alert(
            title="Data Quality",
            message=(
                f"Missing values detected "
                f"({missing_percentage:.1%})."
            ),
        )

    else:

        render_error_alert(
            title="Poor Data Quality",
            message=(
                f"High percentage of missing values "
                               f"({missing_percentage:.1%})."
            ),
        )

def render_empty_state_alert(
    *,
    message: str = "No data available.",
) -> None:
    """
    Display an empty-state notification.
    """

    st.info(
        message,
    )

__all__ = [
    "render_info_alert",
    "render_success_alert",
    "render_warning_alert",
    "render_error_alert",
    "render_exception_alert",
    "render_validation_alert",
    "render_prediction_alert",
    "render_model_status_alert",
    "render_data_quality_alert",
    "render_empty_state_alert",
]