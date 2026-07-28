"""
===============================================================================
Customer Churn Prediction Platform
Layout Components

File        : layout.py
Version     : 1.0

Purpose
-------
Reusable layout primitives for the Streamlit application.

Responsibilities
----------------
• Page headers
• Hero sections
• Section containers
• Dividers
• Empty states
• Responsive layouts
• Footer
===============================================================================
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

def page_header(
    *,
    title: str,
    subtitle: str | None = None,
    icon: str | None = None,
) -> None:
    """
    Render the standard page header.
    """

    heading = f"{icon} {title}" if icon else title

    st.markdown(
        f"""
<div class="page-header">

<h1 class="page-title">{heading}</h1>

{"<p class='page-description'>" + subtitle + "</p>" if subtitle else ""}

</div>
""",
        unsafe_allow_html=True,
    )

def hero(
    *,
    title: str,
    description: str,
) -> None:
    """
    Render the application hero banner.
    """

    st.markdown(
        f"""
<div class="hero">

<h1 class="hero-title">
{title}
</h1>

<p class="hero-subtitle">
{description}
</p>

</div>
""",
        unsafe_allow_html=True,
    )

def section_title(
    title: str,
    description: str | None = None,
) -> None:
    """
    Render a section heading.
    """

    st.markdown(
        f"""
<h2 class="section-title">{title}</h2>

{"<p class='section-subtitle'>" + description + "</p>" if description else ""}
""",
        unsafe_allow_html=True,
    )

def divider() -> None:
    """
    Render a horizontal divider.
    """

    st.markdown(
        "<div class='divider'></div>",
        unsafe_allow_html=True,
    )

def spacer(
    height: int = 24,
) -> None:
    """
    Add vertical spacing.
    """

    st.markdown(
        f"<div style='height:{height}px'></div>",
        unsafe_allow_html=True,
    )
###################

@contextmanager
def section() -> Iterator[None]:
    """
    Card-like content section.

    Example
    -------
    with section():
        st.write(...)
    """

    st.markdown(
        "<div class='section'>",
        unsafe_allow_html=True,
    )

    yield

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

@contextmanager
def card() -> Iterator[None]:
    """
    Generic dashboard card.
    """

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True,
    )

    yield

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

def empty_state(
    *,
    title: str,
    description: str,
    icon: str = "",
) -> None:
    """
    Display an empty-state component.
    """

    st.markdown(
        f"""
<div class="empty-state">

<h2>{icon}</h2>

<h3>{title}</h3>

<p>{description}</p>

</div>
""",
        unsafe_allow_html=True,
    )

def footer() -> None:
    """
    Standard application footer.
    """

    st.markdown(
        """
<div class="footer">

Customer Churn Prediction Platform

<br>

Built with Streamlit • Scikit-learn • SHAP

</div>
""",
        unsafe_allow_html=True,
    )

def two_columns(
    ratio: tuple[int, int] = (1, 1),
):
    """
    Two-column layout.
    """

    return st.columns(ratio)


def three_columns():
    """
    Three-column layout.
    """

    return st.columns(3)


def four_columns():
    """
    Four-column layout.
    """

    return st.columns(4)

def centered_container(
    width: int = 900,
):
    """
    Begin a centered container.

    Useful for forms or authentication pages.
    """

    st.markdown(
        f"""
<div style="
max-width:{width}px;
margin:auto;
">
""",
        unsafe_allow_html=True,
    )


def end_container():
    """
    Close a centered container.
    """

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

__all__ = [
    "page_header",
    "hero",
    "section_title",
    "divider",
    "spacer",
    "section",
    "card",
    "empty_state",
    "footer",
    "two_columns",
    "three_columns",
    "four_columns",
    "centered_container",
    "end_container",
]