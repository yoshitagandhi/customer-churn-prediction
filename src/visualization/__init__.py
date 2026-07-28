"""Plotting and figure-generation utilities.

This package provides reusable, low-level plotting primitives
(`plots.py`), feature-distribution visualizations (`distributions.py`),
and business-oriented churn charts (`business_charts.py`). All figures
are saved as PNGs under `configs.paths.FIGURES_DIR`.
"""

from src.visualization.business_charts import generate_business_charts
from src.visualization.distributions import (
    analyze_categorical_distributions,
    analyze_numerical_distributions,
)
from src.visualization.plots import (
    configure_plot_style,
    plot_correlation_heatmap,
    plot_target_distribution,
)

__all__ = [
    "configure_plot_style",
    "plot_target_distribution",
    "plot_correlation_heatmap",
    "analyze_numerical_distributions",
    "analyze_categorical_distributions",
    "generate_business_charts",
]
