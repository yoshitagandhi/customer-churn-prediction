"""
Customer Churn Prediction Platform
Dashboard Models

Shared data models used throughout the dashboard package.

These models contain no rendering or business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


# =============================================================================
# Dashboard Data
# =============================================================================


@dataclass(slots=True)
class DashboardData:
    """
    Shared dashboard resources.

    Attributes
    ----------
    evaluation
        Evaluation results returned by the evaluation service.

    features
        Validation feature dataset.

    target
        Validation target labels.

    metadata
        Training metadata loaded from cache.
    """

    evaluation: Any

    features: pd.DataFrame

    target: pd.Series

    metadata: dict[str, Any]

__all__ = [
    "DashboardData",
]