"""Session state management.

Wraps ``st.session_state`` behind small, named functions so pages
never poke at session keys directly — keeping the session schema in
one reviewable place.
"""

from typing import Any

import streamlit as st

_PREDICTION_HISTORY_KEY = "prediction_history"
_UPLOADED_BATCH_KEY = "uploaded_batch_dataframe"
_SELECTED_CUSTOMER_KEY = "selected_customer"
_NAV_PAGE_KEY = "nav_page"


def init_session_state() -> None:
    """Initialize every session-state key this app uses, if not already present."""
    st.session_state.setdefault(_PREDICTION_HISTORY_KEY, [])
    st.session_state.setdefault(_UPLOADED_BATCH_KEY, None)
    st.session_state.setdefault(_SELECTED_CUSTOMER_KEY, None)
    st.session_state.setdefault(_NAV_PAGE_KEY, "Dashboard")


def add_prediction_to_history(prediction_record: dict[str, Any]) -> None:
    """Append a single prediction result to this session's history.

    Args:
        prediction_record: A structured prediction result (probability,
            predicted class, risk level, recommended action, etc.).
    """
    history = st.session_state.setdefault(_PREDICTION_HISTORY_KEY, [])
    history.append(prediction_record)


def get_prediction_history() -> list[dict[str, Any]]:
    """Return every prediction made so far in this session.

    Returns:
        A list of prediction result dictionaries, oldest first.
    """
    return st.session_state.get(_PREDICTION_HISTORY_KEY, [])


def set_uploaded_batch(dataframe: Any) -> None:
    """Store an uploaded batch-prediction file's parsed DataFrame.

    Args:
        dataframe: The parsed DataFrame, or None to clear it.
    """
    st.session_state[_UPLOADED_BATCH_KEY] = dataframe


def get_uploaded_batch() -> Any:
    """Return the currently stored uploaded batch DataFrame, if any."""
    return st.session_state.get(_UPLOADED_BATCH_KEY)


def set_selected_customer(customer_data: dict[str, Any] | None) -> None:
    """Store the customer currently selected for explanation/inspection.

    Args:
        customer_data: The selected customer's raw feature values, or
            None to clear the selection.
    """
    st.session_state[_SELECTED_CUSTOMER_KEY] = customer_data


def get_selected_customer() -> dict[str, Any] | None:
    """Return the currently selected customer's data, if any."""
    return st.session_state.get(_SELECTED_CUSTOMER_KEY)


def set_nav_page(page_name: str) -> None:
    """Programmatically switch the active page (e.g., from a dashboard card).

    Args:
        page_name: The display name of the page to navigate to.
    """
    st.session_state[_NAV_PAGE_KEY] = page_name


def get_nav_page() -> str:
    """Return the currently active page name."""
    return st.session_state.get(_NAV_PAGE_KEY, "Dashboard")
