"""
Customer Churn Prediction Platform
Dashboard Loaders

Load and validate all dashboard resources.

Responsibilities
----------------
• Load cached resources
• Validate datasets
• Execute model evaluation
• Build DashboardData object

This module contains no UI rendering.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from configs.logging_config import get_logger

from app.services.evaluation_service import evaluate_model

from app.utils.cache import (
    get_cached_model,
    get_cached_training_metadata,
    get_cached_validation_dataset,
)

from .models import DashboardData

logger = get_logger(__name__)

def load_cached_resources() -> tuple[
    Any,
    pd.DataFrame,
    pd.Series,
    dict[str, Any],
]:
    """
    Load all cached dashboard resources.

    Returns
    -------
    tuple
        (
            model,
            validation_features,
            validation_target,
            training_metadata,
        )
    """

    logger.info("Loading dashboard resources.")

    model = get_cached_model()

    (
        validation_features,
        validation_target,
    ) = get_cached_validation_dataset()

    metadata = get_cached_training_metadata()

    return (
        model,
        validation_features,
        validation_target,
        metadata,
    )

def validate_dashboard_resources(
    *,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """
    Validate dashboard datasets.

    Raises
    ------
    ValueError
        If required dashboard resources are unavailable.
    """

    if features.empty:
        raise ValueError(
            "Validation feature dataset is unavailable."
        )

    if target.empty:
        raise ValueError(
            "Validation target labels are unavailable."
        )

@st.cache_data(show_spinner=False)
def evaluate_dashboard(
    *,
    _model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate the production model.
    """

    logger.info("Running dashboard evaluation.")

    return evaluate_model(
        model=_model,
        features=features,
        target=target,
        include_diagnostics=False,
    )

def load_dashboard_data() -> DashboardData:
    """
    Load all resources required by the dashboard.

    Returns
    -------
    DashboardData
        Fully initialized dashboard object.
    """

    logger.info("Step 1: Loading cached resources...")
    (
        model,
        features,
        target,
        metadata,
    ) = load_cached_resources()

    logger.info(" Cached resources loaded")

    logger.info("Step 2: Validating resources...")
    validate_dashboard_resources(
        features=features,
        target=target,
    )

    logger.info(" Validation completed")

    logger.info("Step 3: Evaluating dashboard...")
    evaluation = evaluate_dashboard(
        _model=model,
        features=features,
        target=target,
    )

    logger.info(" Dashboard evaluation completed")

    return DashboardData(
        evaluation=evaluation,
        features=features,
        target=target,
        metadata=metadata,
    )
__all__ = [
    "load_dashboard_data",
]
