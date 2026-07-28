"""
===============================================================================
Customer Churn Prediction Platform
Reusable Table Components
===============================================================================

Shared table components used across the application.

Responsibilities
----------------
• Display DataFrames consistently
• Apply formatting
• Validate inputs
• Support download-ready tables
• Standardize analytics tables

This module contains presentation logic only.
"""

from __future__ import annotations

from typing import Any, Callable

import json

import pandas as pd
import streamlit as st

DEFAULT_HEIGHT = 400

DEFAULT_COLUMN_CONFIG: dict = {}

def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the supplied dataframe.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected a pandas DataFrame."
        )




def _display_safe_value(value: Any) -> str | None:
    """Return a string or null so a mixed column is safe for Arrow."""
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, default=str)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def make_dataframe_display_safe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns before Streamlit serializes them with Arrow."""
    validate_dataframe(dataframe)
    safe = dataframe.copy()
    for column in safe.select_dtypes(include=["object"]).columns:
        # A pandas ``object`` column can contain strings, numbers, dicts, and
        # nulls. Arrow requires one concrete type per column, so render all
        # non-null entries as text instead of preserving mixed Python scalars.
        safe[column] = safe[column].map(_display_safe_value).astype("string")
    return safe

def get_table_height(
    dataframe: pd.DataFrame,
    default_height: int = DEFAULT_HEIGHT,
) -> int:
    """
    Compute an appropriate table height.
    """

    rows = len(dataframe)

    return min(
        max(
            rows * 35,
            180,
        ),
        default_height,
    )

def format_dataframe(
    dataframe: pd.DataFrame,
    formatter: Callable[[pd.DataFrame], pd.DataFrame]
    | None = None,
) -> pd.DataFrame:
    """
    Apply optional formatting.
    """

    validate_dataframe(
        dataframe,
    )

    if formatter is None:
        return dataframe

    return formatter(
        dataframe.copy(),
    )

def render_dataframe_table(
    dataframe: pd.DataFrame,
    *,
    height: int | None = None,
    column_config: dict | None = None,
    hide_index: bool = True,
    use_container_width: bool = True,
) -> None:
    """
    Render a DataFrame using the application's
    standard styling.
    """

    validate_dataframe(
        dataframe,
    )

    display_frame = make_dataframe_display_safe(dataframe)

    st.dataframe(
        display_frame,
        width="stretch" if use_container_width else "content",
        hide_index=hide_index,
        height=height
        or get_table_height(
            display_frame,
        ),
        column_config=column_config
        or DEFAULT_COLUMN_CONFIG,
    )

def render_styled_table(
    dataframe: pd.DataFrame,
    *,
    formatter: Callable[[pd.DataFrame], pd.DataFrame]
    | None = None,
    **kwargs,
) -> None:
    """
    Render a formatted dataframe.
    """

    formatted = format_dataframe(
        dataframe,
        formatter,
    )

    render_dataframe_table(
        formatted,
        **kwargs,
    )

def render_downloadable_table(
    dataframe: pd.DataFrame,
    *,
    filename: str,
    button_label: str = "Download CSV",
    **kwargs,
) -> None:
    """
    Display a dataframe with an accompanying
    CSV download button.
    """

    validate_dataframe(
        dataframe,
    )

    render_dataframe_table(
        dataframe,
        **kwargs,
    )

    csv = dataframe.to_csv(
        index=False,
    ).encode(
        "utf-8",
    )

    st.download_button(
        label=button_label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )

def render_paginated_table(
    dataframe: pd.DataFrame,
    *,
    page_size: int = 25,
    key: str = "table_page",
) -> None:
    """
    Render a client-side paginated table.
    """

    validate_dataframe(
        dataframe,
    )

    total_rows = len(
        dataframe,
    )

    if total_rows == 0:

        st.info(
            "No records available."
        )

        return

    total_pages = max(
        (
            total_rows
            + page_size
            - 1
        )
        // page_size,
        1,
    )

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=key,
    )

    start = (
        page - 1
    ) * page_size

    end = start + page_size

    render_dataframe_table(
        dataframe.iloc[
            start:end
        ],
    )

    st.caption(
        f"Showing rows {start + 1:,}–"
        f"{min(end, total_rows):,} "
        f"of {total_rows:,}"
    )

def render_model_comparison_table(
    comparison_frame: pd.DataFrame,
) -> None:
    """
    Render the model comparison results.
    """

    render_downloadable_table(
        comparison_frame,
        filename="model_comparison.csv",
        button_label="Download Model Comparison",
    )

def render_feature_importance_table(
    feature_importance: pd.DataFrame,
) -> None:
    """
    Render ranked feature importance.
    """

    render_downloadable_table(
        feature_importance,
        filename="feature_importance.csv",
        button_label="Download Feature Importance",
    )

def render_prediction_table(
    predictions: pd.DataFrame,
) -> None:
    """
    Render batch prediction results.
    """

    render_paginated_table(
        predictions,
        page_size=25,
        key="prediction_table",
    )

def render_dataset_summary_table(
    dataset_summary: pd.DataFrame,
) -> None:
    """
    Render dataset summary statistics.
    """

    render_dataframe_table(
        dataset_summary,
    )

def render_error_table(
    errors: pd.DataFrame,
) -> None:
    """
    Display prediction errors.
    """

    render_paginated_table(
        errors,
        page_size=20,
        key="error_table",
    )

def render_empty_table(
    message: str = "No data available.",
) -> None:
    """
    Display an empty table state.
    """

    st.info(
        message,
    )
    # =============================================================================
# Public API
# =============================================================================

__all__ = [
    "render_dataframe_table",
    "render_styled_table",
    "render_downloadable_table",
    "render_paginated_table",
    "render_model_comparison_table",
    "render_feature_importance_table",
    "render_prediction_table",
    "render_dataset_summary_table",
    "render_error_table",
    "render_empty_table",
]
