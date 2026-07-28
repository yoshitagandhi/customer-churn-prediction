"""
===============================================================================
Customer Churn Prediction Platform
Card Components
===============================================================================

Purpose
-------
Reusable card components shared across the application.

These components provide consistent presentation for business insights,
model summaries, prediction results, warnings, status messages, and
dataset information.

Architecture
------------
Presentation Layer Only

No business logic or machine learning logic belongs in this module.

Used By
-------
• dashboard.py
• evaluation.py
• prediction.py
• batch_prediction.py
===============================================================================
"""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

DEFAULT_ICON = ""

CARD_TYPES = {
    "info",
    "success",
    "warning",
    "error",
}

def validate_card(
    card: dict[str, Any],
) -> None:
    """
    Validate a card definition.
    """

    required = {
        "title",
        "narrative",
    }

    missing = required.difference(card)

    if missing:

        raise ValueError(
            "Card is missing required keys: "
            f"{', '.join(sorted(missing))}"
        )


def render_card_header(
    title: str,
    icon: str = DEFAULT_ICON,
) -> None:
    """
    Render a card header.
    """

    st.markdown(
        f"### {title}"
    )

def render_card_body(
    narrative: str,
) -> None:
    """
    Render the primary narrative.
    """

    st.write(
        narrative,
    )

def render_card_footer(
    recommendations: list[str] | None = None,
) -> None:
    """
    Render optional recommendations.
    """

    if not recommendations:
        return

    st.markdown("**Recommendations**")

    for recommendation in recommendations:

        st.markdown(
            f"- {recommendation}"
        )
        
def render_information_card(
    *,
    title: str,
    narrative: str,
    recommendations: list[str] | None = None,
) -> None:
    """
    Render a generic information card.
    """

    render_card_header(
        title=title,
        icon="ℹ️",
    )

    render_card_body(
        narrative,
    )

    render_card_footer(
        recommendations,
    )


def render_business_insight_card(
    card: dict[str, Any],
) -> None:
    """
    Render a standardized business insight card.

    Parameters
    ----------
    card
        Dictionary containing the card content.

    Required Keys
    -------------
    title
    narrative

    Optional Keys
    -------------
    recommendations
    icon
    """

    validate_card(
        card,
    )

    recommendations = card.get("recommendations") or []
    recommendation_markup = "".join(
        f"<li>{escape(str(item))}</li>" for item in recommendations
    )
    footer = (
        f"<div class=\"insight-recommendations\"><strong>Recommendations</strong>"
        f"<ul>{recommendation_markup}</ul></div>"
        if recommendation_markup
        else ""
    )
    st.markdown(
        f"""
        <section class="business-insight-card">
            <h3>{escape(str(card['title']))}</h3>
            <p>{escape(str(card['narrative']))}</p>
            {footer}
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_success_card(
    *,
    title: str,
    narrative: str,
    recommendations: list[str] | None = None,
) -> None:
    """
    Render a success card.
    """

    st.success(
        title,
    )

    render_card_body(
        narrative,
    )

    render_card_footer(
        recommendations,
    )

def render_warning_card(
    *,
    title: str,
    narrative: str,
    recommendations: list[str] | None = None,
) -> None:
    """
    Render a warning card.
    """

    st.warning(
        title,
    )

    render_card_body(
        narrative,
    )

    render_card_footer(
        recommendations,
    )

def render_error_card(
    *,
    title: str,
    narrative: str,
    recommendations: list[str] | None = None,
) -> None:
    """
    Render an error card.
    """

    st.error(
        title,
    )

    render_card_body(
        narrative,
    )

    render_card_footer(
        recommendations,
    )

def render_model_card(
    *,
    model_name: str,
    version: str,
    training_date: str,
    feature_count: int,
) -> None:
    """
    Display production model information.
    """

    render_card_header(
        title="Production Model",
        icon="",
    )

    st.write(
        f"**Model:** {model_name}"
    )

    st.write(
        f"**Version:** {version}"
    )

    st.write(
        f"**Training Date:** {training_date}"
    )

    st.write(
        f"**Features:** {feature_count:,}"
    )

def render_dataset_card(
    *,
    dataset_name: str,
    samples: int,
    features: int,
    target_column: str,
) -> None:
    """
    Render dataset summary information.
    """

    render_card_header(
        title="Dataset",
        icon="️",
    )

    st.write(
        f"**Dataset:** {dataset_name}"
    )

    st.write(
        f"**Samples:** {samples:,}"
    )

    st.write(
        f"**Features:** {features:,}"
    )

    st.write(
        f"**Target:** {target_column}"
    )

def render_prediction_card(
    *,
    prediction: str,
    probability: float,
    confidence: float,
) -> None:
    """
    Display a prediction summary.
    """

    icon = (
        "️"
        if prediction.lower() == "churn"
        else ""
    )

    render_card_header(
        title="Prediction Result",
        icon=icon,
    )

    st.write(
        f"**Prediction:** {prediction}"
    )

    st.write(
        f"**Probability:** {probability:.2%}"
    )

    st.write(
        f"**Confidence:** {confidence:.2%}"
    )

def render_recommendation_card(
    *,
    title: str,
    recommendations: list[str],
) -> None:
    """
    Display actionable recommendations.
    """

    render_card_header(
        title=title,
        icon="",
    )

    if not recommendations:

        st.info(
            "No recommendations available."
        )

        return

    for recommendation in recommendations:

        st.markdown(
            f"- {recommendation}"
        )

def render_status_card(
    *,
    title: str,
    status: str,
    description: str,
) -> None:
    """
    Render a status overview card.
    """

    status_icons = {
        "healthy": "",
        "warning": "",
        "critical": "",
        "unknown": "",
    }

    icon = status_icons.get(
        status.lower(),
        "",
    )

    render_card_header(
        title=title,
        icon=icon,
    )

    st.write(
        f"**Status:** {status.title()}"
    )

    st.write(
        description,
    )

def render_empty_card(
    *,
    title: str,
    message: str,
) -> None:
    """
    Display an empty-state card.
    """

    render_card_header(
        title=title,
        icon="",
    )

    st.info(
        message,
    )

__all__ = [
    "render_information_card",
    "render_business_insight_card",
    "render_success_card",
    "render_warning_card",
    "render_error_card",
    "render_model_card",
    "render_dataset_card",
    "render_prediction_card",
    "render_recommendation_card",
    "render_status_card",
    "render_empty_card",
]
