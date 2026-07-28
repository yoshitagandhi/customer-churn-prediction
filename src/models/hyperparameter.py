"""Hyperparameter optimization.

Defines a search space per tunable model and runs
``RandomizedSearchCV`` with a reproducible ``StratifiedKFold`` split.
The estimator being tuned is always the *full* preprocessing +
sampling + model pipeline, so resampling happens fresh inside every
cross-validation fold's training split only — never on the fold used
for scoring. This is what makes cross-validation leakage-safe with an
imbalanced-learn pipeline.
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from configs.config import settings
from configs.logging_config import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)

# Search spaces use bare parameter names; tune_model() prefixes them
# with "model__" since the estimator passed to RandomizedSearchCV is
# always a full pipeline with a step named "model".
_SEARCH_SPACES: Final[dict[str, dict[str, Any]]] = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],  # liblinear supports both l1 and l2 penalties
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "xgboost": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
    },
}


@dataclass
class HyperparameterSearchResult:
    """Outcome of a hyperparameter search for one model.

    Attributes:
        model_name: Registered model name that was tuned.
        best_pipeline: The fitted full pipeline (preprocessing +
            sampling + model) using the best parameter combination.
        best_params: The best parameter combination found, with the
            "model__" pipeline prefix stripped for readability.
        best_cv_score: The best cross-validated score achieved.
        cv_folds: Number of cross-validation folds used.
        n_iter: Number of parameter combinations sampled.
        scoring: The scoring metric optimized.
    """

    model_name: str
    best_pipeline: Any
    best_params: dict[str, Any]
    best_cv_score: float
    cv_folds: int
    n_iter: int
    scoring: str


def list_tunable_models() -> tuple[str, ...]:
    """Return the names of every model with a defined search space.

    Returns:
        A tuple of model names.
    """
    return tuple(_SEARCH_SPACES.keys())


def get_search_space(model_name: str) -> dict[str, Any]:
    """Return the hyperparameter search space for a model.

    Args:
        model_name: Registered model name.

    Returns:
        A mapping of parameter name to the values/distribution
        RandomizedSearchCV should sample from.

    Raises:
        ConfigurationError: If no search space is defined for
            ``model_name``.
    """
    if model_name not in _SEARCH_SPACES:
        raise ConfigurationError(
            f"No hyperparameter search space defined for '{model_name}'. "
            f"Tunable models: {list_tunable_models()}"
        )
    return dict(_SEARCH_SPACES[model_name])


def tune_model(
    model_name: str,
    pipeline: Any,
    features_train: pd.DataFrame,
    target_train: pd.Series | np.ndarray,
    n_iter: int = settings.hyperparameter_search_iterations,
    cv_folds: int = settings.cv_folds,
    scoring: str = settings.scoring_metric,
    random_state: int = settings.random_seed,
) -> HyperparameterSearchResult:
    """Run RandomizedSearchCV over a model's search space.

    Args:
        model_name: Registered model name; used to look up its
            search space.
        pipeline: The full, unfitted preprocessing + sampling + model
            pipeline, with the model step named "model".
        features_train: Training features (raw, pre-preprocessing).
        target_train: Training labels, encoded as 0/1.
        n_iter: Number of parameter combinations to sample. Defaults
            to ``settings.hyperparameter_search_iterations``.
        cv_folds: Number of cross-validation folds. Defaults to
            ``settings.cv_folds``.
        scoring: Scoring metric to optimize. Defaults to
            ``settings.scoring_metric``.
        random_state: Random seed for reproducibility. Defaults to
            ``settings.random_seed``.

    Returns:
        A HyperparameterSearchResult with the best fitted pipeline,
        best parameters, and best cross-validated score.
    """
    search_space = get_search_space(model_name)
    prefixed_search_space = {f"model__{key}": value for key, value in search_space.items()}

    cross_validator = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=prefixed_search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cross_validator,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )

    logger.info("Hyperparameter search started for '%s'.", model_name)
    search.fit(features_train, target_train)

    best_params = {
        key.removeprefix("model__"): value for key, value in search.best_params_.items()
    }
    logger.info(
        "Best parameters found for '%s': %s (cv_score=%.4f).",
        model_name,
        best_params,
        search.best_score_,
    )

    return HyperparameterSearchResult(
        model_name=model_name,
        best_pipeline=search.best_estimator_,
        best_params=best_params,
        best_cv_score=round(float(search.best_score_), 4),
        cv_folds=cv_folds,
        n_iter=n_iter,
        scoring=scoring,
    )
