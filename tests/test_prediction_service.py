import pandas as pd


class MockModel:
    classes_ = [0, 1]

    def predict(self, dataframe):
        return [1] * len(dataframe)

    def predict_proba(self, dataframe):
        return [[0.10, 0.90] 
                for _ in range(len(dataframe))
                ]
    
import pytest
import pandas as pd

from app.services.prediction_service import (
    PredictionResult,
    predict_customer,
    predict_batch,
    summarize_predictions,
    _determine_risk_level,
    _recommended_action,
    _prediction_probability,
    ConfigurationError,
    DataValidationError,
)

def test_high_risk():
    assert _determine_risk_level(0.90) == "High"


def test_medium_risk():
    assert _determine_risk_level(0.60) == "Medium"


def test_low_risk():
    assert _determine_risk_level(0.20) == "Low"
    
def test_high_action():
    assert "retention" in _recommended_action("High").lower()


def test_medium_action():
    assert "monitor" in _recommended_action("Medium").lower()


def test_low_action():
    assert "normal" in _recommended_action("Low").lower()


def test_unknown_action():
    assert _recommended_action("ABC") == "Review customer profile."
    
def test_prediction_probability():
    model = MockModel()

    probability = _prediction_probability(
        model,
        1,
        [0.10, 0.90],
    )

    assert probability == 0.90
    
def test_predict_customer():
    model = MockModel()

    customer = {
        "CreditScore": 700
    }

    result = predict_customer(customer, model)
    assert isinstance(result, PredictionResult)
    assert result.predicted_class == "1"
    assert result.risk_level == "High"
    assert result.predicted_probability == 0.9
    assert isinstance(result.customer_frame, pd.DataFrame)
    
def test_predict_batch():
    model = MockModel()

    df = pd.DataFrame({
        "CreditScore":[700,650]
    })

    result = predict_batch(df, model)
    assert "predicted_probability" in result.columns
    assert "risk_level" in result.columns
    assert len(result) == 2
    
def test_summary():

    df = pd.DataFrame({

        "risk_level":[
            "High",
            "Medium",
            "Low",
            "High"
        ]
    })

    summary = summarize_predictions(df)
    assert summary["high_risk"] == 2
    assert summary["medium_risk"] == 1
    assert summary["low_risk"] == 1
    
def test_validate_model_none():
    with pytest.raises(ConfigurationError):
        predict_customer({"CreditScore": 700}, None)
        
class NoPredict:
    def predict_proba(self, dataframe):
        return [[0.2, 0.8]]
    
def test_missing_predict():
    with pytest.raises(ConfigurationError):
        predict_customer({"CreditScore": 700}, NoPredict())
        
class NoPredictProba:
    def predict(self, dataframe):
        return [1]
    
def test_missing_predict_proba():
    with pytest.raises(ConfigurationError):
        predict_customer({"CreditScore": 700}, NoPredictProba())
        
def test_empty_customer():
    with pytest.raises(DataValidationError):
        predict_customer({}, MockModel())
        
def test_empty_dataframe():
    with pytest.raises(DataValidationError):
        predict_batch(pd.DataFrame(), MockModel())
        
class UnknownClassModel:
    classes_ = [0, 1]

    def predict(self, dataframe):
        return [2] * len(dataframe)

    def predict_proba(self, dataframe):
        return [[0.30, 0.70] for _ in range(len(dataframe))]
    
def test_prediction_probability_fallback():
    result = predict_customer(
        {"CreditScore": 700},
        UnknownClassModel(),
    )

    assert result.predicted_probability == 0.7
    
def test_summary_without_risk_level():

    df = pd.DataFrame({
        "CreditScore": [700]
    })

    with pytest.raises(DataValidationError):
        summarize_predictions(df)