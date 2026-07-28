import json

import pandas as pd

from app.services import model_service


def test_export_predictions_csv():

    dataframe = pd.DataFrame(
        {
            "prediction": ["Yes", "No"],
            "probability": [0.91, 0.18],
        }
    )

    result = model_service.export_predictions_csv(
        dataframe
    )

    assert isinstance(result, bytes)

    text = result.decode()

    assert "prediction" in text
    assert "probability" in text
    assert "Yes" in text

def test_export_prediction_json():

    prediction = {
        "predicted_probability": 0.88,
        "predicted_class": "Yes",
    }

    result = model_service.export_prediction_json(
        prediction
    )

    decoded = json.loads(result.decode())

    assert decoded["predicted_probability"] == 0.88
    assert decoded["predicted_class"] == "Yes"

    assert "generated_at" in decoded


def test_export_prediction_markdown_summary():

    prediction = {
        "predicted_probability": 0.91,
        "predicted_class": "Yes",
        "risk_level": "High",
        "recommended_action": "Call customer",
    }

    markdown = (
        model_service
        .export_prediction_markdown_summary(
            prediction
        )
        .decode()
    )

    assert "# Prediction Summary" in markdown
    assert "Predicted probability" in markdown
    assert "Call customer" in markdown


def test_export_prediction_markdown_summary_with_insights():

    prediction = {
        "predicted_probability": 0.91,
        "predicted_class": "Yes",
        "risk_level": "High",
        "recommended_action": "Call customer",
    }

    insights = {
        "narrative": "Customer likely to churn.",
        "recommendations": [
            "Offer discount",
            "Assign account manager",
        ],
    }

    markdown = (
        model_service
        .export_prediction_markdown_summary(
            prediction,
            insights,
        )
        .decode()
    )

    assert "Explanation" in markdown
    assert "Customer likely to churn." in markdown
    assert "Offer discount" in markdown
    assert "Assign account manager" in markdown


def test_export_prediction_markdown_summary_without_recommendations():

    prediction = {
        "predicted_probability": 0.50,
        "predicted_class": "No",
        "risk_level": "Low",
        "recommended_action": "None",
    }

    insights = {
        "narrative": "Low churn risk.",
    }

    markdown = (
        model_service
        .export_prediction_markdown_summary(
            prediction,
            insights,
        )
        .decode()
    )

    assert "Low churn risk." in markdown
    assert "Suggested Business Actions" not in markdown