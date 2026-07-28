import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from configs.config import settings
from src.models import registry
from src.utils.exceptions import ConfigurationError


def test_model_spec_defaults():

    spec = registry.ModelSpec(
        name="demo",
        display_name="Demo Model",
        estimator_factory=lambda: object(),
    )

    assert spec.name == "demo"
    assert spec.display_name == "Demo Model"
    assert spec.tunable is True


def test_model_spec_custom_tunable():

    spec = registry.ModelSpec(
        name="baseline",
        display_name="Baseline",
        estimator_factory=lambda: object(),
        tunable=False,
    )

    assert spec.tunable is False


def test_build_baseline():

    model = registry._build_baseline()

    assert isinstance(model, DummyClassifier)
    assert model.strategy == "most_frequent"


def test_build_logistic_regression():

    model = registry._build_logistic_regression()

    assert isinstance(model, LogisticRegression)
    assert model.random_state == settings.random_seed
    assert model.max_iter == 1000


def test_build_random_forest():

    model = registry._build_random_forest()

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == settings.random_seed
    assert model.n_estimators == 200
    assert model.n_jobs == -1


def test_build_gradient_boosting():

    model = registry._build_gradient_boosting()

    assert isinstance(model, GradientBoostingClassifier)
    assert model.random_state == settings.random_seed


def test_get_models_returns_copy():

    models = registry.get_models()

    assert isinstance(models, dict)

    models.pop("baseline")

    assert "baseline" in registry.get_models()


def test_get_models_contains_expected():

    models = registry.get_models()

    expected = {
        "baseline",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }

    assert expected.issubset(models.keys())


@pytest.mark.parametrize(
    "name",
    [
        "baseline",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    ],
)
def test_get_model_spec(name):

    spec = registry.get_model_spec(name)

    assert spec.name == name


def test_get_model_spec_invalid():

    with pytest.raises(ConfigurationError):

        registry.get_model_spec("does_not_exist")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("baseline", DummyClassifier),
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("gradient_boosting", GradientBoostingClassifier),
    ],
)
def test_build_estimator(name, expected):

    estimator = registry.build_estimator(name)

    assert isinstance(estimator, expected)


def test_build_estimator_invalid():

    with pytest.raises(ConfigurationError):

        registry.build_estimator("invalid_model")


def test_build_xgboost():

    try:
        model = registry._build_xgboost()

        params = model.get_params()

        assert params["random_state"] == settings.random_seed
        assert params["eval_metric"] == "logloss"

    except ImportError:

        pytest.skip("xgboost is not installed")