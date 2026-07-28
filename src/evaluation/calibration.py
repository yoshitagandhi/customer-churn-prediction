"""Probability calibration analysis.

Assesses whether a model's predicted probabilities can be trusted as
probabilities (not just as a ranking signal) — important context for
Milestone 9's threshold optimization.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.visualization.plots import save_figure

logger = get_logger(__name__)

def generate_calibration_curve(
    target_true: np.ndarray,
    target_proba: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    output_dir: Path = FIGURES_DIR,
    filename: str = "calibration_curve.png",
) -> dict[str, Any]:
    """Plot a reliability diagram and compute the Brier score for one model.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.
        model_name: Model name, used in the plot title.
        n_bins: Number of probability bins for the reliability
            diagram.
        output_dir: Directory to save the figure into.
        filename: Output filename.

    Returns:
        A dictionary with the figure path and the Brier score
        (lower is better; 0 is a perfectly calibrated model).
    """
    if len(target_true) == 0:
        raise ValueError("Target labels cannot be empty.")

    if len(target_true) != len(target_proba):
        raise ValueError("Target labels and probabilities must have identical lengths.")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    """Plot a reliability diagram and compute the Brier score for one model.

    Args:
        target_true: Ground-truth labels, encoded as 0/1.
        target_proba: Predicted probability of the positive class.
        model_name: Model name, used in the plot title.
        n_bins: Number of probability bins for the reliability
            diagram.
        output_dir: Directory to save the figure into.
        filename: Output filename.

    Returns:
        A dictionary with the figure path and the Brier score
        (lower is better; 0 is a perfectly calibrated model).
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        target_true, target_proba, n_bins=n_bins, strategy="uniform"
    )
    brier_score = round(float(brier_score_loss(target_true, target_proba)), 4)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(mean_predicted_value, fraction_of_positives, marker="o", label=model_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Observed Frequency of Positives")
    axis.set_title(f"Calibration Curve — {model_name} (Brier score={brier_score})")
    axis.legend(loc="upper left", fontsize=9)

    figure_path = save_figure(figure, filename, output_dir)
    plt.close(figure)
    logger.debug("Calibration curve generated for '%s': Brier score=%s.", model_name, brier_score)

    return {"figure_path": figure_path, "brier_score": brier_score}
