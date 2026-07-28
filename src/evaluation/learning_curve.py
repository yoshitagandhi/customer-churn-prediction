"""Learning curve analysis for the best-performing model.

Refits the already-tuned pipeline at increasing training-set sizes to
diagnose overfitting, underfitting, or data sufficiency. This is not
hyperparameter tuning — the pipeline's configuration (hyperparameters
found in Milestone 6) is reused as-is via ``sklearn.base.clone``,
which resets fitted state without changing configuration.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, learning_curve

from configs.config import settings
from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.evaluation.metrics import compute_metrics
from src.evaluation.calibration import generate_calibration_curve
from src.visualization.plots import save_figure

logger = get_logger(__name__)


def generate_learning_curve(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """Generate a learning curve for a fitted pipeline.

    The supplied pipeline is cloned to ensure learning-curve fitting does not
    modify the original estimator state.
    """

    logger.info("Generating learning curve for %s.", model_name)

    target_encoded = (target == settings.positive_label).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    train_sizes = np.linspace(0.1, 1.0, 5)

    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=clone(pipeline),
        X=features,
        y=target_encoded,
        train_sizes=train_sizes,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        shuffle=False,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes_abs, train_scores_mean, label="Training score")
    ax.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
    )
    ax.plot(train_sizes_abs, valid_scores_mean, label="Cross-validation score")
    ax.fill_between(
        train_sizes_abs,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
    )
    ax.set_title(f"Learning Curve - {model_name}")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC AUC")
    ax.legend(loc="best")
    ax.grid(True)

    figure_path = Path(FIGURES_DIR) / f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
    save_figure(fig, figure_path)
    plt.close(fig)

    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores_mean": train_scores_mean.tolist(),
        "train_scores_std": train_scores_std.tolist(),
        "valid_scores_mean": valid_scores_mean.tolist(),
        "valid_scores_std": valid_scores_std.tolist(),
        "figure_path": str(figure_path),
    }


def evaluate_models(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a single trained production model.

    Used by the Streamlit dashboard.
    """

    logger.info("Evaluating production model.")

    target_encoded = (target == settings.positive_label).astype(int)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics = compute_metrics(
        target_true=target_encoded.to_numpy(),
        target_pred=predictions,
        target_proba=probabilities,
    )

    calibration = generate_calibration_curve(
        target_true=target_encoded.to_numpy(),
        target_proba=probabilities,
        model_name="Production Model",
    )

    learning_curve = generate_learning_curve(
        pipeline=model,
        features=features,
        target=target,
        model_name="Production Model",
    )

    return {
        "metrics": metrics,
        "calibration": calibration,
        "learning_curve": learning_curve,
    }
