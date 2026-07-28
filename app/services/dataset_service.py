"""
===============================================================================
Customer Churn Prediction Platform
Dataset Service

File        : dataset_service.py
Version     : 1.0

Purpose
-------
Provides a thin service layer between the Streamlit UI and the
backend data modules.

Responsibilities
----------------
• Load dataset
• Return dataset summary
• Return dataset profile
• Return dataset preview
• Cache expensive operations

The service does not perform preprocessing, validation,
or profiling itself. Those responsibilities belong to the
backend data package.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data.loader import load_dataset
from src.data.profiler import profile_dataset

@st.cache_data(show_spinner=False)
def get_dataset(
    dataset_path: str | Path,
) -> pd.DataFrame:
    """
    Load the dataset.

    Parameters
    ----------
    dataset_path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
    """

    return load_dataset(dataset_path)

@st.cache_data(show_spinner=False)
def get_dataset_profile(
    dataset_path: str | Path,
):
    """
    Return the backend dataset profile.
    """

    dataset = get_dataset(dataset_path)

    return profile_dataset(dataset)

@st.cache_data(show_spinner=False)
def get_dataset_summary(
    dataset_path: str | Path,
) -> dict[str, Any]:
    """
    Return summary information used by the UI.
    """

    dataset = get_dataset(dataset_path)

    profile = get_dataset_profile(dataset_path)

    return {
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "memory_mb": round(
            dataset.memory_usage(deep=True).sum()
            / (1024 ** 2),
            2,
        ),
        "profile": profile,
    }

@st.cache_data(show_spinner=False)
def get_dataset_preview(
    dataset_path: str | Path,
    rows: int = 10,
) -> pd.DataFrame:
    """
    Return preview rows.
    """

    dataset = get_dataset(dataset_path)

    return dataset.head(rows)

@st.cache_data(show_spinner=False)
def get_column_names(
    dataset_path: str | Path,
) -> list[str]:
    """
    Return dataset column names.
    """

    dataset = get_dataset(dataset_path)

    return list(dataset.columns)

__all__ = [
    "get_dataset",
    "get_dataset_profile",
    "get_dataset_summary",
    "get_dataset_preview",
    "get_column_names",
]