"""
Shared test fixtures for the MLOps project.

Provides reusable fixtures for sample data, trained models, and API clients.
These fixtures are automatically available to all test files.
"""
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


@pytest.fixture
def sample_raw_data():
    """A small sample of raw Telco Churn data for testing."""
    return pd.DataFrame({
        "customerID": ["0001-TEST", "0002-TEST", "0003-TEST", "0004-TEST", "0005-TEST"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 0, 1, 0, 1],
        "Partner": ["Yes", "No", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "Yes", "No"],
        "tenure": [1, 34, 0, 45, 72],
        "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "No", "Yes", "No internet service", "Yes"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service", "Yes"],
        "TechSupport": ["No", "No", "Yes", "No internet service", "Yes"],
        "StreamingTV": ["No", "Yes", "No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No", "Yes", "No internet service", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 104.80],
        "TotalCharges": ["29.85", "1889.5", " ", "1840.75", "7555.6"],
        "Churn": ["Yes", "No", "Yes", "No", "No"],
    })


@pytest.fixture
def sample_clean_data(sample_raw_data):
    """Sample data after cleaning (TotalCharges fixed, Churn encoded)."""
    from data.load import clean_data, encode_target
    df = clean_data(sample_raw_data)
    df = encode_target(df)
    return df


@pytest.fixture
def sample_processed_data(sample_clean_data):
    """Sample data after feature engineering."""
    from data.preprocess import preprocess_for_training
    return preprocess_for_training(sample_clean_data)


@pytest.fixture
def sample_customer_input():
    """Sample customer input for API testing."""
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 958.20,
    }


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from serving.app import app
    return TestClient(app)
