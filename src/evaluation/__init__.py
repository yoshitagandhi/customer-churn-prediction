"""
Evaluation package public API.
"""

from .calibration import generate_calibration_curve
from .comparison import compare_models, identify_best_model
from .learning_curve import generate_learning_curve
from .metrics import compute_metrics
from .report import generate_evaluation_report
from .visualizations import (
    plot_classification_report_heatmap,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from .evaluator import (
    evaluate_model,
    evaluate_models,
)

__all__ = [
    "evaluate_models",
    "evaluate_model", 
    "compute_metrics",
    "compare_models",
    "identify_best_model",
    "generate_calibration_curve",
    "generate_learning_curve",
    "generate_evaluation_report",
    "plot_classification_report_heatmap",
    "plot_confusion_matrix",
    "plot_metric_comparison",
    "plot_precision_recall_curve",
    "plot_roc_curve",
]