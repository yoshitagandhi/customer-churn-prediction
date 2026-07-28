"""
Customer Churn Prediction Platform
Sidebar Components

Reusable sidebar sections shared across the application.

Responsibilities
----------------
• Branding
• Navigation
• Model controls
• Dataset controls
• Export controls
• Developer settings

Presentation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import streamlit as st

DEFAULT_THRESHOLD: Final[float] = 0.50

EXPORT_FORMATS: Final[tuple[str, ...]] = (
    "CSV",
    "Excel",
    "JSON",
)

@dataclass(slots=True)
class SidebarState:
    """
    User selections returned from the sidebar.
    """

    page: str
    model: str
    threshold: float
    dataset: Any
    export_format: str
    debug: bool
    
def render_branding() -> None:
    """
    Render application branding.
    """

    st.sidebar.title(
        "Customer Churn Platform"
    )

    st.sidebar.caption(
        "Production ML Dashboard"
    )

    st.sidebar.divider()
    
def render_navigation(
    pages: list[str],
) -> str:
    """
    Render page navigation.
    """

    return st.sidebar.radio(
        label="Navigation",
        options=pages,
    )
    
def render_model_controls(
    models: list[str],
) -> tuple[str, float, bool]:
    """
    Render model configuration controls.
    """

    model = st.sidebar.selectbox(
        "Model",
        options=models,
    )

    threshold = st.sidebar.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
    )

    reload_requested = st.sidebar.button(
        "Reload Model",
        width="stretch",
    )

    return (
        model,
        threshold,
        reload_requested,
    )
    
def render_dataset_controls() -> tuple[Any, bool]:
    """
    Render dataset controls.
    """

    uploaded = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv"],
    )

    cleared = st.sidebar.button(
        "Clear Dataset",
        width="stretch",
    )

    return uploaded, cleared

def render_export_controls() -> tuple[str, bool]:
    """
    Render export controls.
    """

    export_format = st.sidebar.selectbox(
        "Export Format",
        options=EXPORT_FORMATS,
    )

    export_requested = st.sidebar.button(
        "Export Results",
        width="stretch",
    )

    return (
        export_format,
        export_requested,
    )
    
def render_debug_toggle() -> bool:
    """
    Render debug toggle.
    """

    return st.sidebar.checkbox(
        "Enable debug mode",
        value=False,
    )

def render_about() -> None:
    """
    Render about section.
    """

    st.sidebar.caption(
        "Built for customer churn prediction and model exploration."
    )


def render_sidebar(
    *,
    pages: list[str],
    models: list[str],
) -> SidebarState:
    """
    Render the complete application sidebar.
    """

    render_branding()

    page = render_navigation(pages)

    model, threshold, _ = render_model_controls(models)

    dataset, _ = render_dataset_controls()

    export_format, _ = render_export_controls()

    debug = render_debug_toggle()

    render_about()

    return SidebarState(
        page=page,
        model=model,
        threshold=threshold,
        dataset=dataset,
        export_format=export_format,
        debug=debug,
    )
    
__all__ = [
    "SidebarState",
    "render_sidebar",
]
    
