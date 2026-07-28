import pandas as pd

from app.utils.validators import (
    validate_customer_input,
    validate_batch_dataframe,
)

def test_valid_customer(valid_customer):
    errors = validate_customer_input(valid_customer)
    assert errors == []
    
def test_missing_required_field(valid_customer):
    valid_customer.pop("gender")
    errors = validate_customer_input(valid_customer)
    assert any("gender" in error for error in errors)
    
def test_invalid_gender(valid_customer):
    valid_customer["gender"] = "ABC"
    errors = validate_customer_input(valid_customer)
    assert any("gender" in error for error in errors)
    
def test_invalid_numeric(valid_customer):
    valid_customer["MonthlyCharges"] = "abc"
    errors = validate_customer_input(valid_customer)
    assert any("MonthlyCharges" in error for error in errors)
    
def test_numeric_out_of_range(valid_customer):
    valid_customer["tenure"] = 1000
    errors = validate_customer_input(valid_customer)
    assert any("tenure" in error for error in errors)
    
def test_invalid_senior(valid_customer):
    valid_customer["SeniorCitizen"] = 5
    errors = validate_customer_input(valid_customer)
    assert any("SeniorCitizen" in error for error in errors)
    
def test_empty_dataframe():
    df = pd.DataFrame()
    errors = validate_batch_dataframe(df)
    assert errors
    
def test_missing_columns():
    df = pd.DataFrame({

        "gender": ["Male"]

    })

    errors = validate_batch_dataframe(df)
    assert any("Missing required" in error for error in errors)
    
def test_invalid_target(valid_dataframe):
    valid_dataframe["Churn"] = ["ABC"]
    errors = validate_batch_dataframe(valid_dataframe)
    assert any("unexpected" in error.lower() for error in errors)
    
def test_valid_dataframe(valid_dataframe):
    errors = validate_batch_dataframe(valid_dataframe)
    assert errors == []
    
def test_numeric_field_none(valid_customer):
    valid_customer["MonthlyCharges"] = None

    errors = validate_customer_input(valid_customer)

    assert any("MonthlyCharges" in error for error in errors)
    
