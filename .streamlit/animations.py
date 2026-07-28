"""
===============================================================================
Customer Churn Prediction Platform
Animation Utilities

File        : animations.py
Version     : 1.0

Purpose
-------
Animation helpers for the Streamlit UI.

Responsibilities
----------------
• Loading overlays
• Progress animations
• Skeleton placeholders
• Toast notifications
• Fade-in containers
• Animated status messages

The animation styles themselves are defined in styles.css.
===============================================================================
"""

from __future__ import annotations

from contextlib import contextmanager
from time import sleep
from typing import Iterator

import streamlit as st


###############################################################################
# LOADING OVERLAY
###############################################################################

@contextmanager
def loading(
    message: str = "Loading...",
) -> Iterator[None]:
    """
    Display a loading overlay while executing a block.

    Example
    -------
    with loading("Training model..."):
        train_model()
    """

    placeholder = st.empty()

    placeholder.markdown(
        f"""
<div class="loading-overlay">

<div class="spinner"></div>

<p>{message}</p>

</div>
""",
        unsafe_allow_html=True,
    )

    try:
        yield

    finally:
        placeholder.empty()


###############################################################################
# SKELETON PLACEHOLDER
###############################################################################

def skeleton(
    *,
    rows: int = 4,
) -> None:
    """
    Display animated skeleton placeholders.
    """

    html = []

    for _ in range(rows):

        html.append(
            """
<div class="skeleton skeleton-text"></div>
"""
        )

    st.markdown(
        f"""
<div class="card">

<div class="skeleton skeleton-title"></div>

{''.join(html)}

</div>
""",
        unsafe_allow_html=True,
    )


###############################################################################
# FADE-IN CONTAINER
###############################################################################

@contextmanager
def fade_in() -> Iterator[None]:
    """
    Wrap UI inside a fade-in animation.
    """

    st.markdown(
        '<div class="fade-in">',
        unsafe_allow_html=True,
    )

    try:
        yield

    finally:
        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


###############################################################################
# TOAST
###############################################################################

def toast(
    message: str,
    *,
    icon: str = "",
) -> None:
    """
    Display a toast notification.
    """

    st.toast(message, icon=icon)


###############################################################################
# SUCCESS ANIMATION
###############################################################################

def success(
    message: str,
) -> None:
    """
    Display a success message.
    """

    st.success(message)


###############################################################################
# ERROR ANIMATION
###############################################################################

def error(
    message: str,
) -> None:
    """
    Display an error message.
    """

    st.error(message)


###############################################################################
# WARNING ANIMATION
###############################################################################

def warning(
    message: str,
) -> None:
    """
    Display a warning message.
    """

    st.warning(message)


###############################################################################
# INFO ANIMATION
###############################################################################

def info(
    message: str,
) -> None:
    """
    Display an informational message.
    """

    st.info(message)


###############################################################################
# ANIMATED PROGRESS
###############################################################################

def animated_progress(
    *,
    total_steps: int,
    delay: float = 0.02,
) -> None:
    """
    Animate a progress bar.

    Parameters
    ----------
    total_steps
        Number of animation steps.

    delay
        Delay between updates.
    """

    progress = st.progress(0)

    if total_steps <= 0:
        progress.empty()
        return

    for step in range(total_steps + 1):

        progress.progress(step / total_steps)

        sleep(delay)

    progress.empty()


###############################################################################
# STATUS TRANSITION
###############################################################################

def status_transition(
    *,
    status: str,
    message: str,
) -> None:
    """
    Display a status message using the appropriate style.
    """

    status = status.lower()

    handlers = {
        "success": success,
        "error": error,
        "warning": warning,
        "info": info,
    }

    handler = handlers.get(status, info)
    handler(message)


###############################################################################
# PUBLIC EXPORTS
###############################################################################

__all__ = [
    "loading",
    "skeleton",
    "fade_in",
    "toast",
    "success",
    "error",
    "warning",
    "info",
    "animated_progress",
    "status_transition",
]