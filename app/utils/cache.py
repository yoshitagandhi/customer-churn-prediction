"""
===============================================================================
Customer Churn Prediction Platform
Centralized Cache Manager
===============================================================================

Purpose
-------
Provides a single location for all Streamlit caching.

Responsibilities
----------------
• Load and cache trained models
• Load training metadata
• Load threshold configuration
• Load evaluation datasets
• Load experiment history
• Cache SHAP explainers
• Cache SHAP background data

Notes
-----
• Prediction results are NEVER cached.
• Only immutable artifacts are cached.
• All Streamlit cache decorators live here.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from configs.config import settings
from configs.logging_config import get_logger

from src.data import load_dataset
from src.preprocessing import clean_dataset
from src.models import (
    load_model,
    load_training_metadata,
)
from src.threshold import load_threshold_config
from src.explainability import (
    get_processed_features,
    load_explainer,
)

logger = get_logger(__name__)

@st.cache_resource(show_spinner="Loading trained model...")
def get_cached_model(
    model_path: str = str(settings.best_model_path),
) -> Any:
    """
    Load the trained production model.
    """

    logger.info("Loading trained model...")

    model = load_model(Path(model_path))

    logger.info("Model loaded successfully.")

    return model

@st.cache_data(show_spinner=False)
def get_cached_training_metadata(
    metadata_path: str = str(settings.training_metadata_path),
) -> dict[str, Any]:
    """
    Load cached training metadata.
    """

    logger.info("Loading training metadata.")

    metadata = load_training_metadata(
        Path(metadata_path),
    )

    return metadata

@st.cache_data(show_spinner=False)
def get_cached_threshold_config(
    config_path: str = str(settings.threshold_config_path),
) -> dict[str, Any]:
    """
    Load optimized threshold configuration.
    """

    logger.info("Loading threshold configuration.")

    return load_threshold_config(
        Path(config_path),
    )

@st.cache_data(show_spinner="Loading dataset...")
def get_cached_dataset() -> pd.DataFrame:
    """
    Load cleaned dataset.
    """

    raw = load_dataset()

    cleaned = clean_dataset(raw)

    logger.info("Dataset cached.")

    return cleaned

@st.cache_data(show_spinner=False)
def get_cached_validation_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Return feature matrix and target labels.

    Used by evaluation page.
    """

    dataset = get_cached_dataset()

    target = dataset[settings.target_column]

    features = dataset.drop(
        columns=[settings.target_column],
        errors="ignore",
    )

    return features, target

@st.cache_data(show_spinner=False)
def get_cached_background_sample(
    sample_size: int = 200,
) -> pd.DataFrame:
    """
    Load SHAP background dataset.
    """

    dataset = get_cached_dataset()

    features = dataset.drop(
        columns=[settings.target_column],
        errors="ignore",
    )

    if len(features) > sample_size:

        features = features.sample(
            n=sample_size,
            random_state=settings.random_seed,
        )

    return features.reset_index(drop=True)

@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_cached_explainer(
    _pipeline: Any,
    background_key: str = "default",
) -> Any:
    """
    Build cached SHAP explainer.
    """

    background = get_cached_background_sample()

    processed = get_processed_features(
        _pipeline,
        background,
    )

    explainer = load_explainer(
        _pipeline,
        processed,
    )

    logger.info(
        "SHAP explainer cached (%s).",
        background_key,
    )

    return explainer

@st.cache_data(show_spinner=False)
def get_cached_experiment_records() -> list[dict[str, Any]]:
    """
    Return experiment history.

    Training metadata should contain
    'experiment_records'.
    """

    metadata = get_cached_training_metadata()

    return metadata.get(
        "experiment_records",
        [],
    )

@st.cache_data(show_spinner=False)
def get_cached_feature_metadata() -> dict[str, Any]:
    """
    Return feature metadata.
    """

    metadata = get_cached_training_metadata()

    return metadata.get(
        "feature_metadata",
        {},
    )

def clear_all_cache() -> None:
    """
    Clear all Streamlit caches.
    """

    st.cache_data.clear()

    st.cache_resource.clear()

    logger.info("Application cache cleared.")

__all__ = [
    "get_cached_model",
    "get_cached_dataset",
    "get_cached_validation_dataset",
    "get_cached_training_metadata",
    "get_cached_threshold_config",
    "get_cached_background_sample",
    "get_cached_explainer",
    "get_cached_experiment_records",
    "get_cached_feature_metadata",
    "clear_all_cache",
]