"""
===============================================================================
Layout Components

Reusable layout helpers shared throughout the application.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

DEFAULT_LAYOUT = "wide"

DEFAULT_ICON = "CP"

def configure_page(
    *,
    title: str = "Customer Churn Prediction",
    icon: str = DEFAULT_ICON,
    layout: str = DEFAULT_LAYOUT,
) -> None:
    """
    Configure the Streamlit page.
    """

    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
    )

def render_page_header(
    title: str,
    description: str | None = None,
) -> None:
    """
    Render page heading.
    """

    st.title(title)

    if description:

        st.caption(
            description,
        )

    st.divider()
    
def bordered_container():
    """
    Standard bordered container.
    """
    return st.container(border=True)


def metric_container():
    """
    Container for KPI metrics.
    """
    return st.container(border=True)


def chart_container():
    """
    Container for visualizations.
    """
    return st.container(border=True)


def card_container():
    """
    Container for insight cards.
    """
    return st.container(border=True)

def two_column_layout():
    """
    Two equal columns.
    """
    return st.columns(2)


def three_column_layout():
    """
    Three equal columns.
    """
    return st.columns(3)


def four_column_layout():
    """
    Four equal columns.
    """
    return st.columns(4)


def responsive_columns(
    ratios: Sequence[int],
):
    """
    Custom responsive layout.
    """
    return st.columns(
        list(ratios),
    )


def information_expander(
    title: str = "Information",
):
    """
    Standard information expander.
    """
    return st.expander(
        title,
        expanded=False,
    )


def technical_details_expander():
    """
    Technical details expander.
    """
    return st.expander(
        "Technical Details",
    )


def advanced_options_expander():
    """
    Advanced options.
    """
    return st.expander("Advanced Options", )
    
    __all__ = [
    "configure_page",
    "render_page_header",
    "bordered_container",
    "metric_container",
    "chart_container",
    "card_container",
    "two_column_layout",
    "three_column_layout",
    "four_column_layout",
    "responsive_columns",
    "information_expander",
    "technical_details_expander",
    "advanced_options_expander",
]