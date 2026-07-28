"""
Session State Manager

Centralized session state management for the Customer Churn
Prediction Platform.

The SessionManager is the single source of truth for all
Streamlit session state interactions.

Responsibilities
----------------
• Session initialization
• Typed state access
• State mutation
• Application reset
• Default values
• Session consistency

The rest of the application should never access
st.session_state directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import streamlit as st


@dataclass(frozen=True)
class SessionKeys:
    """Canonical session-state keys."""

    MODEL: str = "model"
    METRICS: str = "metrics"
    PREDICTION: str = "prediction"
    BATCH_PREDICTIONS: str = "batch_predictions"
    FEATURE_IMPORTANCE: str = "feature_importance"
    SELECTED_MODEL: str = "selected_model"
    SIDEBAR_EXPANDED: str = "sidebar_expanded"


SESSION_DEFAULTS: Final[dict[str, Any]] = {
    SessionKeys.MODEL: None,
    SessionKeys.METRICS: None,
    SessionKeys.PREDICTION: None,
    SessionKeys.BATCH_PREDICTIONS: None,
    SessionKeys.FEATURE_IMPORTANCE: None,
    SessionKeys.SELECTED_MODEL: None,
    SessionKeys.SIDEBAR_EXPANDED: True,
}

class SessionManager:
    """
    Centralized manager for Streamlit session state.

    This class is the only component that should read from or
    write to ``st.session_state``. All pages, components, and
    actions should interact with session state exclusively
    through this API.
    """

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize all managed session values.

        Existing values are preserved while any missing keys
        are populated using SESSION_DEFAULTS.
        """

        for key, default in SESSION_DEFAULTS.items():
            st.session_state.setdefault(key, default)

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Retrieve a value from session state.
        """

        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Store a value in session state.
        """

        st.session_state[key] = value

    @staticmethod
    def exists(key: str) -> bool:
        """
        Determine whether a session key exists.
        """

        return key in st.session_state

    @staticmethod
    def remove(key: str) -> None:
        """
        Remove a single session key.

        Missing keys are ignored.
        """

        st.session_state.pop(key, None)

    @classmethod
    def reset(cls) -> None:
        """
        Remove every managed session key.

        This does not clear unrelated Streamlit state.
        """

        for key in SESSION_DEFAULTS:
            cls.remove(key)

    @classmethod
    def reset_predictions(cls) -> None:
        """
        Remove prediction-related state.
        """

        cls.set(SessionKeys.PREDICTION, None)
        cls.set(SessionKeys.BATCH_PREDICTIONS, None)

    @classmethod
    def clear_model(cls) -> None:
        """
        Remove the active model.
        """

        cls.set(SessionKeys.MODEL, None)

    @classmethod
    def clear_metrics(cls) -> None:
        """
        Remove stored evaluation metrics.
        """

        cls.set(SessionKeys.METRICS, None)

    @classmethod
    def clear_feature_importance(cls) -> None:
        """
        Remove stored feature importance.
        """

        cls.set(SessionKeys.FEATURE_IMPORTANCE, None)

    @classmethod
    def reset_analysis(cls) -> None:
        """
        Reset all prediction and evaluation artifacts while
        preserving application configuration.
        """

        cls.reset_predictions()
        cls.clear_metrics()
        cls.clear_feature_importance()

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        """
        Return the managed session state as a dictionary.

        Useful for debugging or exporting state.
        """

        return {
            key: cls.get(key)
            for key in SESSION_DEFAULTS
        }
        
# =============================================================================
# Model State
# =============================================================================

    @classmethod
    def set_model(cls, model: Any) -> None:
        """Store the active model."""

        cls.set(SessionKeys.MODEL, model)

    @classmethod
    def get_model(cls) -> Any:
        """Return the active model."""

        return cls.get(SessionKeys.MODEL)

    @classmethod
    def has_model(cls) -> bool:
        """Return True if a model is currently loaded."""

        return cls.get_model() is not None

    @classmethod
    def save_prediction(
        cls,
        prediction: dict[str, Any],
    ) -> None:
        """Store the latest prediction."""

        cls.set(
            SessionKeys.PREDICTION,
            prediction,
        )

    @classmethod
    def get_prediction(
        cls,
    ) -> dict[str, Any] | None:
        """Return the latest prediction."""

        return cls.get(
            SessionKeys.PREDICTION,
        )

    @classmethod
    def save_batch_predictions(
        cls,
        predictions: Any,
    ) -> None:
        """Store batch prediction results."""

        cls.set(
            SessionKeys.BATCH_PREDICTIONS,
            predictions,
        )

    @classmethod
    def get_batch_predictions(
        cls,
    ) -> Any:
        """Return batch prediction results."""

        return cls.get(
            SessionKeys.BATCH_PREDICTIONS,
        )

    @classmethod
    def save_metrics(
        cls,
        metrics: dict[str, Any],
    ) -> None:
        """Store evaluation metrics."""

        cls.set(
            SessionKeys.METRICS,
            metrics,
        )

    @classmethod
    def get_metrics(
        cls,
    ) -> dict[str, Any] | None:
        """Return evaluation metrics."""

        return cls.get(
            SessionKeys.METRICS,
        )

    @classmethod
    def save_feature_importance(
        cls,
        dataframe: Any,
    ) -> None:
        """Store feature importance."""

        cls.set(
            SessionKeys.FEATURE_IMPORTANCE,
            dataframe,
        )

    @classmethod
    def get_feature_importance(
        cls,
    ) -> Any:
        """Return feature importance."""

        return cls.get(
            SessionKeys.FEATURE_IMPORTANCE,
        )

    @classmethod
    def save_sidebar_state(
        cls,
        expanded: bool,
    ) -> None:
        """Persist sidebar state."""

        cls.set(
            SessionKeys.SIDEBAR_EXPANDED,
            expanded,
        )

    @classmethod
    def get_sidebar_state(
        cls,
    ) -> bool:
        """Return sidebar expansion state."""

        return cls.get(
            SessionKeys.SIDEBAR_EXPANDED,
            True,
        )

    @classmethod
    def save_selected_model(
        cls,
        model_name: str,
    ) -> None:
        """Persist selected model."""

        cls.set(
            SessionKeys.SELECTED_MODEL,
            model_name,
        )

    @classmethod
    def get_selected_model(
        cls,
    ) -> str | None:
        """Return selected model."""

        return cls.get(
            SessionKeys.SELECTED_MODEL,
        )

__all__ = [
    "SessionKeys",
    "SessionManager",
]     