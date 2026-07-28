"""
Dashboard Constants
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# =============================================================================
# Page
# =============================================================================

PAGE_TITLE: Final = "Executive Dashboard"

PAGE_DESCRIPTION: Final = (
    "Executive overview of customer churn prediction, "
    "model performance, business insights, and analytics."
)

EXCELLENT_SCORE: Final = 0.90
GOOD_SCORE: Final = 0.80
WARNING_SCORE: Final = 0.70

OVERVIEW_TITLE: Final = "Executive Overview"

KPI_TITLE: Final = "Model Performance"

BUSINESS_TITLE: Final = "Business Insights"

ANALYTICS_TITLE: Final = "Analytics"

NAVIGATION_TITLE: Final = "Explore Platform"

@dataclass(frozen=True, slots=True)
class NavigationCard:

    title: str
    description: str
    page: str


NAVIGATION_CARDS: Final = (
    NavigationCard(
        "Prediction",
        "Predict customer churn.",
        "Prediction",
    ),
    NavigationCard(
        "Batch Prediction",
        "Upload customer datasets.",
        "Batch Prediction",
    ),
    NavigationCard(
        "Model Performance",
        "View evaluation metrics.",
        "Model Performance",
    ),
    NavigationCard(
        "Explainability",
        "Understand model decisions.",
        "Explainability",
    ),
    NavigationCard(
        "Evaluation",
        "Inspect detailed reports.",
        "Evaluation",
    ),
)