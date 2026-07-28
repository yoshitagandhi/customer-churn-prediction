"""
Customer Churn Prediction Platform
Dashboard Navigation

Navigation hub for the Executive Dashboard.

Responsibilities
----------------
• Display available application pages
• Explain each module
• Provide a quick platform overview

This module intentionally does not implement page routing.
"""

from __future__ import annotations

import streamlit as st

from .constants import (
    NAVIGATION_CARDS,
    NAVIGATION_TITLE,
)

def render_navigation_card(
    *,
    title: str,
    description: str,
    page: str,
) -> None:
    """
    Render a navigation information card.
    """

    st.markdown(
        f"""
        <article class="explore-card">
            <h3>{title}</h3>
            <p>{description}</p>
            <span>OPEN {page.upper()} →</span>
        </article>
        """,
        unsafe_allow_html=True,
    )

def render_navigation_hub() -> None:
    """
    Render the platform navigation hub.
    """

    with st.container():
        st.markdown('<section class="explore-panel"><div class="attention-heading"><i>↗</i> Explore Platform</div><p>Jump into the tools that turn customer signals into confident retention actions.</p>', unsafe_allow_html=True)
        columns = st.columns(2)
        for index, card in enumerate(NAVIGATION_CARDS):
            with columns[index % 2]:
                render_navigation_card(title=card.title, description=card.description, page=card.page)
        st.markdown("</section>", unsafe_allow_html=True)

__all__ = [
    "render_navigation_hub",
]
