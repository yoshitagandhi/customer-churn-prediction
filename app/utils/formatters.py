"""
===============================================================================
Customer Churn Prediction Platform
Formatting Utilities

File        : formatters.py
Version     : 1.0

Purpose
-------
Provides reusable formatting helpers for the Streamlit application.

Responsibilities
----------------
• Probability formatting
• Percentage formatting
• Currency formatting
• Metric formatting
• Date & time formatting
• Duration formatting
• File size formatting
• Label formatting

Notes
-----
• Pure formatting only.
• No business logic.
• No Streamlit imports.
• No model operations.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

DEFAULT_DECIMAL_PLACES: Final[int] = 2

UNKNOWN_VALUE: Final[str] = "N/A"

BYTES_PER_KB: Final[int] = 1024
BYTES_PER_MB: Final[int] = 1024**2
BYTES_PER_GB: Final[int] = 1024**3

def format_probability(
    probability: float | None,
    decimals: int = DEFAULT_DECIMAL_PLACES,
) -> str:
    """
    Format probability as a percentage.

    Example
    -------
    0.8731 -> 87.31%
    """

    if probability is None:
        return UNKNOWN_VALUE

    return f"{probability * 100:.{decimals}f}%"

def format_percentage(
    value: float | int | None,
    decimals: int = DEFAULT_DECIMAL_PLACES,
) -> str:
    """
    Format percentage value.

    Example
    -------
    95.2345 -> 95.23%
    """

    if value is None:
        return UNKNOWN_VALUE

    return f"{value:.{decimals}f}%"

def format_number(
    value: float | int | None,
    decimals: int = DEFAULT_DECIMAL_PLACES,
) -> str:
    """
    Format numeric value with separators.

    Example
    -------
    1250000 -> 1,250,000.00
    """

    if value is None:
        return UNKNOWN_VALUE

    return f"{value:,.{decimals}f}"

def format_integer(
    value: int | None,
) -> str:
    """
    Format integer with commas.
    """

    if value is None:
        return UNKNOWN_VALUE

    return f"{value:,}"

def format_currency(
    value: float | int | None,
    symbol: str = "₹",
    decimals: int = DEFAULT_DECIMAL_PLACES,
) -> str:
    """
    Format currency.

    Example
    -------
    5234.4 -> ₹5,234.40
    """

    if value is None:
        return UNKNOWN_VALUE

    return f"{symbol}{value:,.{decimals}f}"

def format_metric(
    value: float | None,
    decimals: int = 4,
) -> str:
    """
    Format machine learning metric.
    """

    if value is None:
        return UNKNOWN_VALUE

    return f"{value:.{decimals}f}"

def format_accuracy(
    accuracy: float | None,
) -> str:
    """
    Format accuracy score.
    """

    return format_probability(accuracy)

def format_auc(
    auc: float | None,
) -> str:
    """
    Format ROC-AUC.
    """

    return format_metric(auc)

def format_f1_score(
    score: float | None,
) -> str:
    """
    Format F1 score.
    """

def format_datetime(
    value: datetime | str | None,
    pattern: str = "%d %b %Y %H:%M",
) -> str:
    """
    Format datetime.

    Accepts datetime object or ISO string.
    """

    if value is None:
        return UNKNOWN_VALUE

    if isinstance(value, str):
        value = datetime.fromisoformat(value)

    return value.strftime(pattern)

def format_date(
    value: datetime | str | None,
) -> str:
    """
    Format date.
    """

    return format_datetime(
        value,
        "%d %b %Y",
    )

def format_time(
    value: datetime | str | None,
) -> str:
    """
    Format time.
    """

    return format_datetime(
        value,
        "%H:%M:%S",
    )

def format_duration(
    seconds: float | int | None,
) -> str:
    """
    Convert seconds into readable duration.

    Example
    -------
    125 -> 2m 5s
    """

    if seconds is None:
        return UNKNOWN_VALUE

    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"

def format_file_size(
    size_bytes: int | None,
) -> str:
    """
    Format bytes into KB / MB / GB.
    """

    if size_bytes is None:
        return UNKNOWN_VALUE

    if size_bytes >= BYTES_PER_GB:
        return f"{size_bytes / BYTES_PER_GB:.2f} GB"

    if size_bytes >= BYTES_PER_MB:
        return f"{size_bytes / BYTES_PER_MB:.2f} MB"

    if size_bytes >= BYTES_PER_KB:
        return f"{size_bytes / BYTES_PER_KB:.2f} KB"

    return f"{size_bytes} B"

def format_label(
    label: str | None,
) -> str:
    """
    Convert snake_case or kebab-case into title case.

    Example
    -------
    roc_auc -> Roc Auc
    """

    if not label:
        return UNKNOWN_VALUE

    return (
        label.replace("_", " ")
        .replace("-", " ")
        .title()
    )

def format_boolean(
    value: bool | None,
) -> str:
    """
    Format boolean.

    Example
    -------
    True -> Yes
    False -> No
    """

    if value is None:
        return UNKNOWN_VALUE

    return "Yes" if value else "No"

def safe_display(
    value: Any,
) -> str:
    """
    Safely convert values for UI display.
    """

    if value is None:
        return UNKNOWN_VALUE

    if value == "":
        return UNKNOWN_VALUE

    return str(value)

__all__ = [
    "format_probability",
    "format_percentage",
    "format_number",
    "format_integer",
    "format_currency",
    "format_metric",
    "format_accuracy",
    "format_auc",
    "format_f1_score",
    "format_datetime",
    "format_date",
    "format_time",
    "format_duration",
    "format_file_size",
    "format_label",
    "format_boolean",
    "safe_display",
]