"""
===============================================================================
Customer Churn Prediction Platform
Icons

File        : icons.py
Author      : Yoshita Gandhi
Version     : 1.0

Purpose
-------
Centralized icon definitions for the Streamlit UI.

Using a single source of truth ensures consistency across all pages and
makes it easy to replace emoji icons with SVG or icon fonts in the future.
===============================================================================
"""

from __future__ import annotations

from typing import Final

APP: Final[str] = ""
HOME: Final[str] = ""
DASHBOARD: Final[str] = ""
SETTINGS: Final[str] = "️"
ABOUT: Final[str] = "ℹ️"

DATASET: Final[str] = "️"
UPLOAD: Final[str] = ""
DOWNLOAD: Final[str] = "⬇️"
DATABASE: Final[str] = "️"
TABLE: Final[str] = ""

MODEL: Final[str] = ""
TRAINING: Final[str] = "️"
PREDICTION: Final[str] = ""
FEATURES: Final[str] = ""
TARGET: Final[str] = ""
EXPERIMENT: Final[str] = ""

METRICS: Final[str] = ""
ROC: Final[str] = ""
CONFUSION_MATRIX: Final[str] = ""
CALIBRATION: Final[str] = "️"
LEARNING_CURVE: Final[str] = ""
COMPARISON: Final[str] = "️"

SHAP: Final[str] = ""
FEATURE_IMPORTANCE: Final[str] = "⭐"
INSIGHTS: Final[str] = ""
REPORT: Final[str] = ""

CUSTOMER: Final[str] = ""
PROFILE: Final[str] = ""
COMPANY: Final[str] = ""

SUCCESS: Final[str] = ""
WARNING: Final[str] = "️"
ERROR: Final[str] = ""
INFO: Final[str] = "ℹ️"
LOADING: Final[str] = "⏳"

NEXT: Final[str] = "️"
BACK: Final[str] = "⬅️"
REFRESH: Final[str] = ""
SEARCH: Final[str] = ""
FILTER: Final[str] = ""

LOGIN: Final[str] = ""
LOGOUT: Final[str] = ""
USER: Final[str] = ""
ADMIN: Final[str] = ""

COPY: Final[str] = ""
SAVE: Final[str] = ""
DELETE: Final[str] = "️"
EDIT: Final[str] = "️"
EXPORT: Final[str] = ""
IMPORT: Final[str] = ""

__all__ = [
    name
    for name in globals()
    if name.isupper()
]