"""Central model registry.

Every model available for training is registered here exactly once,
as a name plus a factory function that returns a fresh, unfitted
estimator with sensible default parameters. Adding a future model
(LightGBM, CatBoost, SVM, ...) only requires adding one factory
function and one registry entry — no other module needs to change.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from configs.config import settings
from configs.logging_config import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single registered model.

    Attributes:
        name: Unique registry key (e.g., "xgboost").
        display_name: Human-readable name used in reports.
        estimator_factory: Zero-argument callable returning a fresh,
            unfitted estimator with sensible default parameters.
        tunable: Whether a hyperparameter search space is defined for
            this model in :mod:`src.models.hyperparameter`. Models
            without one are trained once with their default
            parameters as a comparison baseline.
    """

    name: str
    display_name: str
    estimator_factory: Callable[[], Any]
    tunable: bool = True


def _build_baseline() -> DummyClassifier:
    """Build the baseline model: always predicts the majority class."""
    return DummyClassifier(strategy="most_frequent")


def _build_logistic_regression() -> LogisticRegression:
    """Build a Logistic Regression classifier with sensible defaults."""
    return LogisticRegression(random_state=settings.random_seed, max_iter=1000)


def _build_random_forest() -> RandomForestClassifier:
    """Build a Random Forest classifier with sensible defaults."""
    return RandomForestClassifier(
        random_state=settings.random_seed, n_estimators=200, n_jobs=-1
    )


def _build_gradient_boosting() -> GradientBoostingClassifier:
    """Build a Gradient Boosting classifier with sensible defaults."""
    return GradientBoostingClassifier(random_state=settings.random_seed)


def _build_xgboost() -> Any:
    """Build an XGBoost classifier with sensible defaults.

    Raises:
        ImportError: If the ``xgboost`` package is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "The 'xgboost' package is required for the XGBoost model. "
            "Install it via `pip install xgboost`."
        ) from exc

    return XGBClassifier(
        random_state=settings.random_seed,
        eval_metric="logloss",
        n_jobs=-1,
    )


_REGISTRY: Final[dict[str, ModelSpec]] = {
    "baseline": ModelSpec(
        name="baseline",
        display_name="Baseline (Most Frequent Class)",
        estimator_factory=_build_baseline,
        tunable=False,
    ),
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        estimator_factory=_build_logistic_regression,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        estimator_factory=_build_random_forest,
    ),
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        display_name="Gradient Boosting",
        estimator_factory=_build_gradient_boosting,
        tunable=False,
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        estimator_factory=_build_xgboost,
    ),
}


def get_models() -> dict[str, ModelSpec]:
    """Return every registered model specification.

    Returns:
        A copy of the model registry, keyed by model name.
    """
    return dict(_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a single model's specification by name.

    Args:
        name: Registered model name.

    Returns:
        The matching ModelSpec.

    Raises:
        ConfigurationError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise ConfigurationError(f"Unknown model '{name}'. Registered models: {list(_REGISTRY)}")
    return _REGISTRY[name]


def build_estimator(name: str) -> Any:
    """Build a fresh, unfitted estimator instance for a registered model.

    Args:
        name: Registered model name.

    Returns:
        A new, unfitted estimator instance.

    Raises:
        ConfigurationError: If ``name`` is not registered.
        ImportError: If the model's required package is not installed.
    """
    return get_model_spec(name).estimator_factory()
