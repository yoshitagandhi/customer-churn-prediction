from app.services import explanation_service
import pandas as pd
from src.utils.exceptions import (
    ConfigurationError,
    DataValidationError,
)

def test_validate_model_none():
    try:
        explanation_service._validate_model(None)
        assert False
    except ConfigurationError:
        assert True


def test_validate_explainer_none():
    try:
        explanation_service._validate_explainer(None)
        assert False
    except ConfigurationError:
        assert True


def test_validate_customer_data_empty():
    try:
        explanation_service._validate_customer_data({})
        assert False
    except DataValidationError:
        assert True


def test_explain_customer_prediction(monkeypatch):

    model = object()
    explainer = object()

    customer = {
        "FeatureA": 1,
    }

    monkeypatch.setattr(
        explanation_service,
        "backend_explain_prediction",
        lambda **kwargs: {"prediction": "Yes"},
    )

    monkeypatch.setattr(
        explanation_service,
        "generate_business_insights",
        lambda prediction: {"risk": "High"},
    )

    result = explanation_service.explain_customer_prediction(
        model,
        explainer,
        customer,
    )

    assert result.prediction == {"prediction": "Yes"}
    assert result.business_insights == {"risk": "High"}
    assert result.waterfall_path is None


def test_generate_waterfall(monkeypatch):

    model = object()
    explainer = object()

    customer = {
        "FeatureA": 1,
    }

    monkeypatch.setattr(
        explanation_service,
        "get_processed_features",
        lambda model, frame: "processed",
    )

    monkeypatch.setattr(
        explanation_service,
        "compute_shap_values",
        lambda explainer, processed: "values",
    )

    monkeypatch.setattr(
        explanation_service,
        "plot_shap_waterfall",
        lambda **kwargs: "figure.png",
    )

    result = explanation_service.generate_waterfall_figure(
        model,
        explainer,
        customer,
    )

    assert result == "figure.png"

def test_explain_with_visualization(monkeypatch):

    explanation = explanation_service.ExplanationResult(
        prediction="prediction",
        business_insights="insights",
    )

    monkeypatch.setattr(
        explanation_service,
        "explain_customer_prediction",
        lambda *args, **kwargs: explanation,
    )

    monkeypatch.setattr(
        explanation_service,
        "generate_waterfall_figure",
        lambda *args, **kwargs: "waterfall.png",
    )

    result = explanation_service.explain_with_visualization(
        object(),
        object(),
        {"FeatureA": 1},
    )

    assert result.waterfall_path == "waterfall.png"


def test_explain_customer_prediction_with_scenarios():

    class DummyModel:
        def predict_proba(self, features):
            probability = 0.1 + 0.2 * float(features.iloc[0]["FeatureA"])
            return [[1 - probability, probability]]

    result = explanation_service.explain_customer_prediction_with_scenarios(
        DummyModel(),
        {"FeatureA": 1},
        pd.DataFrame({"FeatureA": [0, 0, 0]}),
    )

    assert result.prediction["explanation_method"] == "Scenario analysis"
    assert result.prediction["top_positive_contributors"][0]["feature"] == "FeatureA"

def test_explanation_result_defaults():

    result = explanation_service.ExplanationResult(
        prediction="prediction",
        business_insights="business",
    )

    assert result.prediction == "prediction"
    assert result.business_insights == "business"
    assert result.waterfall_path is None
