"""
===============================================================================
Customer Churn Prediction Platform
Theme Management

File        : theme.py
Author      : Yoshita Gandhi
Version     : 1.0

Purpose
-------
Centralized Streamlit theme initialization.

Responsibilities
----------------
• Configure Streamlit page settings.
• Load the global CSS stylesheet.
• Inject optional custom HTML into <head>.
• Provide reusable helpers for page initialization.

This module should be called exactly once at the top of every Streamlit page.

Example
-------
from .theme import initialize_theme

initialize_theme(
    page_title="Customer Churn Prediction",
    page_icon="",
)
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import streamlit as st

_STREAMLIT_DIR: Final[Path] = Path(__file__).resolve().parent

_CSS_FILE: Final[Path] = _STREAMLIT_DIR / "styles.css"

def _read_css() -> str:
    """
    Read the global stylesheet.

    Returns
    -------
    str
        Complete CSS stylesheet.

    Raises
    ------
    FileNotFoundError
        If styles.css cannot be found.
    """

    if not _CSS_FILE.exists():
        raise FileNotFoundError(
            f"Unable to locate stylesheet:\n{_CSS_FILE}"
        )

    return _CSS_FILE.read_text(
        encoding="utf-8"
    )


def _inject_css() -> None:
    """
    Inject the stylesheet into Streamlit.
    """

    css = _read_css()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )

def initialize_theme(
    *,
    page_title: str,
    page_icon: str = "",
    layout: str = "wide",
    sidebar_state: str = "expanded",
) -> None:
    """
    Initialize the application theme.

    Parameters
    ----------
    page_title
        Browser page title.

    page_icon
        Emoji or favicon.

    layout
        Streamlit layout.

    sidebar_state
        Initial sidebar state.
    """

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )

    _inject_css()


def load_css() -> None:
    """
    Reload the global stylesheet.

    Useful after hot reload during development.
    """

    _inject_css()

def inject_html(html: str) -> None:
    """
    Render arbitrary HTML.

    Parameters
    ----------
    html
        Raw HTML fragment.
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )

def page_break(
    height: int = 24,
) -> None:
    """
    Insert vertical spacing.

    Parameters
    ----------
    height
        Height in pixels.
    """

    st.markdown(
        f"<div style='height:{height}px'></div>",
        unsafe_allow_html=True,
    )


def horizontal_rule() -> None:
    """
    Render a divider.
    """

    st.markdown("<hr>", unsafe_allow_html=True)

__all__ = [
    "initialize_theme",
    "load_css",
    "inject_html",
    "page_break",
    "horizontal_rule",
]