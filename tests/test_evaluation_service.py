from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from app.services import evaluation_service
from src.utils.exceptions import (
    ConfigurationError,
    DataValidationError,
)


def test_validate_model_success():
    evaluation_service._validate_model(object())


def test_validate_model_none():
    with pytest.raises(ConfigurationError):
        evaluation_service._validate_model(None)


def test_validate_features_success():
    df = pd.DataFrame({"A": [1]})
    evaluation_service._validate_features(df)


def test_validate_features_empty():
    with pytest.raises(DataValidationError):
        evaluation_service._validate_features(pd.DataFrame())


def test_validate_target_success():
    target = pd.Series([0, 1])
    evaluation_service._validate_target(target)


def test_validate_target_empty():
    with pytest.raises(DataValidationError):
        evaluation_service._validate_target(pd.Series(dtype=int))


def test_encode_target_boolean():

    target = pd.Series([True, False])

    result = evaluation_service._encode_target(
        target,
        True,
    )

    assert result.tolist() == [1, 0]


def test_encode_target_numeric():

    target = pd.Series([1, 0, 1])

    result = evaluation_service._encode_target(
        target,
        1,
    )

    assert result.tolist() == [1, 0, 1]


def test_encode_target_string():

    target = pd.Series(
        ["Yes", "No", "Yes"]
    )

    result = evaluation_service._encode_target(
        target,
        "Yes",
    )

    assert result.tolist() == [1, 0, 1]

def test_evaluate_model(monkeypatch):

    class DummyModel:

        def predict(self, X):
            return np.array([1, 0])

        def predict_proba(self, X):
            return np.array(
            [
                [0.1, 0.9],
                [0.8, 0.2],
            ]
        )

    monkeypatch.setattr(
        evaluation_service,
        "compute_metrics",
        lambda **kwargs: {"accuracy": 0.95},
    )

    monkeypatch.setattr(
        evaluation_service,
        "generate_calibration_curve",
        lambda **kwargs: "calibration.png",
    )

    monkeypatch.setattr(
        evaluation_service,
        "generate_learning_curve",
        lambda **kwargs: "learning.png",
    )

    features = pd.DataFrame({"A": [1, 2]})
    target = pd.Series(["Yes", "No"])

    result = evaluation_service.evaluate_model(
        DummyModel(),
        features,
        target,
    )

    assert result["metrics"]["accuracy"] == 0.95
    assert result["predictions"]["prediction"].tolist() == [1, 0]
    assert result["predictions"]["churn_probability"].tolist() == [0.9, 0.2]
    assert result["calibration"] == "calibration.png"
    assert result["learning_curve"] == "learning.png"

def test_compare_models(monkeypatch):

    comparison = SimpleNamespace(
        best_model="Random Forest",
        comparison_table=pd.DataFrame({"A": [1]}),
        ranking=["Random Forest"],
        selection_reason="Highest ROC",
    )

    monkeypatch.setattr(
        evaluation_service,
        "backend_compare_models",
        lambda records: comparison,
    )

    result = evaluation_service.compare_models(
        ["record"]
    )

    assert result.best_model == "Random Forest"
    assert result.selection_reason == "Highest ROC"


def test_compare_models_empty():

    with pytest.raises(DataValidationError):
        evaluation_service.compare_models([])


def test_generate_evaluation_report(monkeypatch):

    monkeypatch.setattr(
        evaluation_service,
        "backend_generate_report",
        lambda **kwargs: "report.pdf",
    )

    evaluation = evaluation_service.EvaluationResult(
        metrics={},
        calibration_curve_path=None,
        learning_curve_path=None,
    )

    result = evaluation_service.generate_evaluation_report(
        evaluation,
        "reports",
    )

    assert result == "report.pdf"


def test_get_learning_curve(monkeypatch):

    monkeypatch.setattr(
        evaluation_service,
        "generate_learning_curve",
        lambda **kwargs: "learning.png",
    )

    result = evaluation_service.get_learning_curve(
        model=object(),
        features=pd.DataFrame({"A": [1]}),
        target=pd.Series([1]),
    )

    assert result == "learning.png"


def test_get_calibration_curve(monkeypatch):

    monkeypatch.setattr(
        evaluation_service,
        "generate_calibration_curve",
        lambda **kwargs: "calibration.png",
    )

    result = evaluation_service.get_calibration_curve(
        model=object(),
        features=pd.DataFrame({"A": [1]}),
        target=pd.Series([1]),
    )

    assert result == "calibration.png"


def test_evaluation_result():

    result = evaluation_service.EvaluationResult(
        metrics={"accuracy": 1},
        calibration_curve_path="cal.png",
        learning_curve_path="learn.png",
    )

    assert result.metrics["accuracy"] == 1
    assert result.calibration_curve_path == "cal.png"


def test_comparison_result():

    result = evaluation_service.ComparisonResult(
        best_model="RF",
        comparison_table=pd.DataFrame(),
        ranking=["RF"],
        selection_reason="Best",
    )

    assert result.best_model == "RF"
    assert result.ranking == ["RF"]
