"""
===============================================================================
Customer Churn Prediction Platform
Application State Manager

File: state.py

Purpose
-------
Centralized management of Streamlit session state.

Responsibilities
----------------
• Initialize application state
• Store UI state
• Store prediction state
• Store evaluation state
• Store uploaded datasets
• Store cached page results
• Reset logical state groups

Notes
-----
• This module is the ONLY place that should directly access
  st.session_state.
• No prediction logic.
• No model loading.
• No business logic.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd
import streamlit as st

@dataclass(slots=True)
class AppState:
    """Default application state."""

    current_page: str = "Dashboard"

    sidebar_expanded: bool = True

    theme: str = "light"

    prediction_result: Any = None

    uploaded_dataframe: pd.DataFrame | None = None

    prediction_dataframe: pd.DataFrame | None = None

    selected_customer: int | None = None

    explanation_result: Any = None

    shap_values: Any = None

    evaluation_result: Any = None

    comparison_result: Any = None

    learning_curve: Any = None

    calibration_curve: Any = None

    training_status: str = "Idle"

    training_metadata: dict[str, Any] = field(default_factory=dict)

    generated_report: Any = None

    exported_file: bytes | None = None

    notifications: list[str] = field(default_factory=list)


DEFAULT_STATE: Final = AppState()

_STATE_KEY: Final[str] = "_app_state"

def initialize_state() -> None:
    """
    Initialize application state.

    Safe to call on every page.
    """

    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = AppState()

def get_state() -> AppState:
    """
    Return application state.
    """

    initialize_state()

    return st.session_state[_STATE_KEY]

def set_state(state: AppState) -> None:
    """
    Replace application state.
    """
    st.session_state[_STATE_KEY] = state

def get_value(name: str) -> Any:
    """
    Retrieve a field from AppState.
    """

    return getattr(get_state(), name)

def set_value(name: str, value: Any) -> None:
    """
    Update a field inside AppState.
    """
    setattr(get_state(), name, value)

def get_prediction_result() -> Any:
    return get_state().prediction_result


def set_prediction_result(result: Any) -> None:
    get_state().prediction_result = result

def reset_prediction_state() -> None:
    state = get_state()

    state.prediction_result = None
    state.prediction_dataframe = None
    state.uploaded_dataframe = None
    state.selected_customer = None

def reset_explainability_state() -> None:
    state = get_state()

    state.explanation_result = None
    state.shap_values = None

def reset_evaluation_state() -> None:
    state = get_state()

    state.evaluation_result = None
    state.comparison_result = None
    state.learning_curve = None
    state.calibration_curve = None

def reset_training_state() -> None:
    state = get_state()

    state.training_status = "Idle"
    state.training_metadata.clear()

def add_notification(message: str) -> None:
    get_state().notifications.append(message)

def clear_notifications() -> None:
    get_state().notifications.clear()

def clear_session() -> None:
    """
    Completely reset application session.
    """

    st.session_state.clear()

    initialize_state()

__all__ = [
    "AppState",
    "initialize_state",
    "get_state",
    "set_state",
    "get_value",
    "set_value",
    "get_prediction_result",
    "set_prediction_result",
    "reset_prediction_state",
    "reset_training_state",
    "reset_evaluation_state",
    "reset_explainability_state",
    "add_notification",
    "clear_notifications",
    "clear_session",
]