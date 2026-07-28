"""
Customer Churn Prediction Platform
Application Actions

Coordinates user interactions between the Streamlit presentation
layer and backend services.

Responsibilities
----------------
• Execute user actions
• Validate user inputs
• Coordinate backend services
• Update SessionManager
• Handle application errors

This module intentionally contains no machine learning,
preprocessing, or business logic.
"""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

import streamlit as st

from .session import SessionManager

PredictionCallable: TypeAlias = Callable[
    [dict[str, Any]],
    Any,
]

BatchPredictionCallable: TypeAlias = Callable[
    [Any],
    Any,
]

ModelLoader: TypeAlias = Callable[
    [],
    Any,
]

DatasetValidator: TypeAlias = Callable[
    [Any],
    bool,
]

DatasetPreprocessor: TypeAlias = Callable[
    [Any],
    Any,
]

Exporter: TypeAlias = Callable[
    ...,
    Any,
]

def execute_action(
    action: Callable[..., Any],
    *args: Any,
    on_success: Callable[[Any], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute an application action.

    Provides centralized exception handling while allowing
    optional success and error callbacks.
    """

    try:

        result = action(
            *args,
            **kwargs,
        )

        if on_success is not None:

            on_success(result)

        return result

    except Exception as exc:

        if on_error is not None:

            on_error(exc)

        st.error("The requested action could not be completed.")

        return None

def validate_prediction_input(
    input_data: dict[str, Any],
) -> bool:
    """
    Validate prediction request.
    """

    return bool(input_data)


def validate_batch_input(
    dataframe: Any,
) -> bool:
    """
    Validate uploaded dataframe.
    """

    if dataframe is None:
        return False

    if getattr(
        dataframe,
        "empty",
        False,
    ):
        return False

    return True

def run_prediction(
    *,
    predictor: PredictionCallable,
    input_data: dict[str, Any],
) -> Any:
    """
    Execute a single customer prediction.

    The prediction result is automatically persisted
    using the SessionManager.
    """

    if not validate_prediction_input(input_data):
        return None

    return execute_action(
        predictor,
        input_data,
        on_success=SessionManager.save_prediction,
    )

def run_batch_prediction(
    *,
    predictor: BatchPredictionCallable,
    dataframe: Any,
) -> Any:
    """
    Execute batch prediction for a dataset.

    Results are automatically stored in session state
    through the SessionManager.
    """

    if not validate_batch_input(dataframe):
        return None

    return execute_action(
        predictor,
        dataframe,
        on_success=SessionManager.save_batch_predictions,
    )

def clear_prediction_action() -> None:
    """
    Remove all stored prediction results.
    """

    SessionManager.reset_predictions()

def load_model_action(
    *,
    loader: ModelLoader,
) -> Any:
    """
    Load the production model.

    On successful loading the model is persisted
    through the SessionManager.
    """

    return execute_action(
        loader,
        on_success=SessionManager.set_model,
    )


def reload_model_action(
    *,
    loader: ModelLoader,
) -> Any:
    """
    Reload the active production model.

    Existing model state is cleared before loading
    a fresh instance.
    """

    SessionManager.clear_model()

    return load_model_action(
        loader=loader,
    )


def unload_model_action() -> None:
    """
    Remove the active model.
    """

    SessionManager.clear_model()


def select_model_action(
    model_name: str,
) -> None:
    """
    Persist the selected model name.

    This represents the user's model selection
    rather than the loaded model instance.
    """

    SessionManager.save_selected_model(
        model_name,
    )

def upload_dataset_action(
    uploaded_file: Any,
) -> Any:
    """
    Return the uploaded dataset.

    Parsing remains the responsibility of the
    backend ingestion pipeline.
    """

    return uploaded_file


def validate_dataset_action(
    *,
    validator: DatasetValidator,
    dataset: Any,
) -> bool:
    """
    Validate an uploaded dataset.
    """

    result = execute_action(
        validator,
        dataset,
    )

    return bool(result)


def preprocess_dataset_action(
    *,
    preprocessor: DatasetPreprocessor,
    dataset: Any,
) -> Any:
    """
    Execute dataset preprocessing.

    Feature engineering remains inside the backend
    preprocessing service.
    """

    return execute_action(
        preprocessor,
        dataset,
    )

def export_results_action(
    exporter: Exporter,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Export prediction results using the supplied exporter.

    The exporter is responsible for serialization,
    file generation, and formatting.
    """

    return execute_action(
        exporter,
        *args,
        **kwargs,
    )

def reset_application_action() -> None:
    """
    Reset all managed application state.

    This removes predictions, metrics,
    feature importance, selected model,
    and the currently loaded model.
    """

    SessionManager.reset()


def clear_cache_action() -> None:
    """
    Clear Streamlit caches.

    Session state is intentionally preserved.
    """

    st.cache_data.clear()
    st.cache_resource.clear()


def refresh_application_action() -> None:
    """
    Refresh application state.

    Clears cached resources while preserving
    the current user session.
    """

    clear_cache_action()

__all__ = [

    # Executor
    "execute_action",

    # Validation
    "validate_prediction_input",
    "validate_batch_input",

    # Prediction
    "run_prediction",
    "run_batch_prediction",
    "clear_prediction_action",

    # Model
    "load_model_action",
    "reload_model_action",
    "unload_model_action",
    "select_model_action",

    # Dataset
    "upload_dataset_action",
    "validate_dataset_action",
    "preprocess_dataset_action",

    # Export
    "export_results_action",

    # Application
    "reset_application_action",
    "clear_cache_action",
    "refresh_application_action",
]