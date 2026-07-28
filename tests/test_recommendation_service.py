from types import SimpleNamespace

import pandas as pd

from app.services import recommendation_service

def fake_high_risk(probability):
    return SimpleNamespace(
        name="High",
        action="Call customer immediately",
    )


def fake_low_risk(probability):
    return SimpleNamespace(
        name="Low",
        action="Continue normal engagement",
    )


def test_generate_recommendation_above_threshold(
    monkeypatch,
    threshold_config,
):
    monkeypatch.setattr(
        recommendation_service,
        "classify_risk",
        fake_high_risk,
    )

    result = recommendation_service.generate_recommendation(
        0.80,
        threshold_config,
    )

    assert result["predicted_class"] == "Yes"
    assert result["predicted_probability"] == 0.80
    assert result["threshold_used"] == 0.5
    assert result["risk_level"] == "High"
    assert result["recommended_action"] == "Call customer immediately"


def test_generate_recommendation_below_threshold(
    monkeypatch,
    threshold_config,
):
    monkeypatch.setattr(
        recommendation_service,
        "classify_risk",
        fake_low_risk,
    )

    result = recommendation_service.generate_recommendation(
        0.20,
        threshold_config,
    )

    assert result["predicted_class"] == "No"
    assert result["predicted_probability"] == 0.20
    assert result["threshold_used"] == 0.5
    assert result["risk_level"] == "Low"
    assert result["recommended_action"] == "Continue normal engagement"


def test_custom_threshold(monkeypatch):
    monkeypatch.setattr(
        recommendation_service,
        "classify_risk",
        fake_high_risk,
    )

    config = {
        "optimal_threshold": 0.90,
    }

    result = recommendation_service.generate_recommendation(
        0.80,
        config,
    )

    assert result["predicted_class"] == "No"
    assert result["threshold_used"] == 0.90


def test_probability_preserved(
    monkeypatch,
    threshold_config,
):
    monkeypatch.setattr(
        recommendation_service,
        "classify_risk",
        fake_high_risk,
    )

    probability = 0.733

    result = recommendation_service.generate_recommendation(
        probability,
        threshold_config,
    )

    assert result["predicted_probability"] == probability

def test_generate_batch_recommendations(
    monkeypatch,
    threshold_config,
):
    monkeypatch.setattr(
        recommendation_service,
        "classify_risk",
        fake_high_risk,
    )

    dataframe = pd.DataFrame(
        {
            "predicted_probability": [
                0.20,
                0.80,
            ]
        }
    )

    result = recommendation_service.generate_batch_recommendations(
        dataframe,
        threshold_config,
    )

    assert len(result) == 2

    assert "predicted_class" in result.columns
    assert "risk_level" in result.columns
    assert "recommended_action" in result.columns

    assert list(result["predicted_class"]) == [
        "No",
        "Yes",
    ]


def test_original_dataframe_not_modified(
    monkeypatch,
    threshold_config,
):
    monkeypatch.setattr(
        recommendation_service,
        "classify_risk",
        fake_high_risk,
    )

    original = pd.DataFrame(
        {
            "predicted_probability": [0.80]
        }
    )

    recommendation_service.generate_batch_recommendations(
        original,
        threshold_config,
    )

    assert "predicted_class" not in original.columns
    assert "risk_level" not in original.columns
    assert "recommended_action" not in original.columns