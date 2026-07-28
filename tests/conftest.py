import pandas as pd
import pytest

@pytest.fixture
def valid_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 700,
    }


@pytest.fixture
def valid_dataframe(valid_customer):
    return pd.DataFrame([valid_customer])

@pytest.fixture
def threshold_config():
    return {
        "optimal_threshold": 0.5,
    }

@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "FeatureA": [1, 2, 3],
            "FeatureB": [4, 5, 6],
            "Churn": [0, 1, 0],
        }
    )


@pytest.fixture
def large_dataset():
    return pd.DataFrame(
        {
            "FeatureA": list(range(500)),
            "FeatureB": list(range(500)),
            "Churn": [0] * 500,
        }
    )

@pytest.fixture
def sample_metadata():
    return {
        "experiment_records": [
            {
                "model": "Random Forest",
                "accuracy": 0.91,
            }
        ],
        "feature_metadata": {
            "FeatureA": "numeric",
            "FeatureB": "numeric",
        },
    }