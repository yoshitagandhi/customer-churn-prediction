"""
===============================================================================
Customer Churn Prediction Platform
Hero Components
===============================================================================

Reusable hero components for application pages.

Responsibilities
----------------
• Page introduction
• Quick statistics
• Primary actions
• Feature highlights

Presentation layer only.
===============================================================================
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# =============================================================================
# Hero Header
# =============================================================================


def render_hero_title(
    title: str,
) -> None:
    """
    Render page title.
    """

    st.title(title)

def render_hero_description(
    description: str,
) -> None:
    """
    Render page description.
    """

    st.caption(description)
    
def render_hero_badges(
    badges: list[str],
) -> None:
    """
    Display hero badges.
    """

    if not badges:
        return

    columns = st.columns(len(badges))

    for column, badge in zip(columns, badges):

        column.info(badge)

def render_hero_actions(
    actions: list[str],
) -> str | None:
    """
    Render hero action buttons.
    """

    if not actions:
        return None

    columns = st.columns(len(actions))

    for column, action in zip(columns, actions):

        if column.button(
            action,
            width="stretch",
        ):
            return action

    return None

def render_quick_metric(
    label: str,
    value: Any,
) -> None:
    """
    Display quick metric.
    """

    st.metric(
        label,
        value,
    )

def render_quick_stats(
    metrics: dict[str, Any],
) -> None:
    """
    Render quick metrics.
    """

    columns = st.columns(
        len(metrics)
    )

    for column, (label, value) in zip(
        columns,
        metrics.items(),
    ):

        with column:

            render_quick_metric(
                label,
                value,
            )

def render_feature_cards(
    features: list[str],
) -> None:
    """
    Display feature highlights.
    """

    st.subheader(
        "Platform Features"
    )

    for feature in features:

        st.success(feature)


def render_business_value(
    message: str,
) -> None:
    """
    Display business value.
    """

    st.info(message)
    
def render_hero(
    *,
    title: str,
    description: str,
    badges: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    actions: list[str] | None = None,
) -> str | None:
    """
    Render the complete hero section.
    """

    render_hero_title(title)

    render_hero_description(description)

    if badges:

        render_hero_badges(badges)

    if metrics:

        render_quick_stats(metrics)

    if actions:

        return render_hero_actions(actions)

    return None

__all__ = [
    "render_hero",
    "render_hero_title",
    "render_hero_description",
    "render_hero_badges",
    "render_hero_actions",
    "render_quick_metric",
    "render_quick_stats",
    "render_feature_cards",
    "render_business_value",
]