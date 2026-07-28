"""
Customer Churn Prediction Platform
Dashboard Formatters

Shared formatting utilities for dashboard rendering.

This module contains presentation helpers only.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


# =============================================================================
# Number Formatting
# =============================================================================


def format_percentage(
    value: float | None,
    decimals: int = 2,
) -> str:
    """
    Format a decimal value as a percentage.

    Examples
    --------
    0.9234 -> 92.34%
    """

    if value is None:
        return "N/A"

    return f"{value:.{decimals}%}"


def format_number(
    value: int | float | None,
) -> str:
    """
    Format numeric values using thousands separators.
    """

    if value is None:
        return "N/A"

    if isinstance(value, int):
        return f"{value:,}"

    return f"{value:,.2f}"


def format_compact_number(
    value: int | float | None,
) -> str:
    """
    Format large values into compact notation.

    Examples
    --------
    1250 -> 1.2K
    1250000 -> 1.2M
    """

    if value is None:
        return "N/A"

    absolute = abs(value)

    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"

    return format_number(value)

def format_datetime(
    value: datetime | date | str | None,
    fmt: str = "%d %b %Y",
) -> str:
    """
    Format supported date values.
    """

    if value is None:
        return "N/A"

    if isinstance(value, str):
        return value

    return value.strftime(fmt)

def format_title(
    value: str,
) -> str:
    """
    Convert text into title case.
    """

    return value.replace("_", " ").title()


def format_label(
    value: str,
) -> str:
    """
    Convert snake_case labels into readable labels.
    """

    return value.replace("_", " ").capitalize()

_STATUS_ICONS = {
    "excellent": "",
    "good": "",
    "healthy": "",
    "stable": "",
    "warning": "",
    "monitor": "",
    "critical": "",
    "error": "",
}


def format_status(
    status: str,
) -> str:
    """
    Format a status with an icon.

    Examples
    --------
    Excellent ->  Excellent
    """

    icon = _STATUS_ICONS.get(
        status.lower(),
        "",
    )

    return f"{icon} {status}"

def format_value(
    value: Any,
) -> str:
    """
    Generic value formatter.
    """

    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return format_number(value)

    return str(value)

__all__ = [
    "format_compact_number",
    "format_datetime",
    "format_label",
    "format_number",
    "format_percentage",
    "format_status",
    "format_title",
    "format_value",
]