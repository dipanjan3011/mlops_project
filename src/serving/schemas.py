"""
Pydantic v2 request/response schemas for the prediction API.

These schemas validate incoming requests and document the API response format.
All fields match the Telco Customer Churn dataset columns.
"""
from pydantic import BaseModel, Field
from typing import Optional


class CustomerInput(BaseModel):
    """Input schema for a single customer prediction request.

    All fields correspond to the raw Telco Churn dataset columns.
    Fields are typed to match the original data types.
    """
    customerID: str = Field(default="unknown", description="Unique customer identifier")
    gender: str = Field(description="Customer gender", examples=["Male", "Female"])
    SeniorCitizen: int = Field(description="Whether the customer is a senior citizen (0 or 1)", ge=0, le=1)
    Partner: str = Field(description="Whether the customer has a partner", examples=["Yes", "No"])
    Dependents: str = Field(description="Whether the customer has dependents", examples=["Yes", "No"])
    tenure: int = Field(description="Number of months the customer has stayed", ge=0, le=72)
    PhoneService: str = Field(description="Whether the customer has phone service", examples=["Yes", "No"])
    MultipleLines: str = Field(description="Whether the customer has multiple lines", examples=["Yes", "No", "No phone service"])
    InternetService: str = Field(description="Customer's internet service provider", examples=["DSL", "Fiber optic", "No"])
    OnlineSecurity: str = Field(description="Whether the customer has online security", examples=["Yes", "No", "No internet service"])
    OnlineBackup: str = Field(description="Whether the customer has online backup", examples=["Yes", "No", "No internet service"])
    DeviceProtection: str = Field(description="Whether the customer has device protection", examples=["Yes", "No", "No internet service"])
    TechSupport: str = Field(description="Whether the customer has tech support", examples=["Yes", "No", "No internet service"])
    StreamingTV: str = Field(description="Whether the customer has streaming TV", examples=["Yes", "No", "No internet service"])
    StreamingMovies: str = Field(description="Whether the customer has streaming movies", examples=["Yes", "No", "No internet service"])
    Contract: str = Field(description="The contract term", examples=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: str = Field(description="Whether the customer has paperless billing", examples=["Yes", "No"])
    PaymentMethod: str = Field(description="The customer's payment method", examples=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges: float = Field(description="The amount charged monthly", ge=0)
    TotalCharges: float = Field(description="The total amount charged", ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customerID": "7590-VHVEG",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


class CustomerIDInput(BaseModel):
    """Input schema for a Feast-powered prediction — only needs the customer ID."""
    customerID: str = Field(description="Unique customer identifier to look up in the feature store")


class PredictionOutput(BaseModel):
    """Response schema for a single prediction."""
    customerID: str = Field(description="Customer identifier from the request")
    churn_prediction: int = Field(description="Binary prediction: 1=will churn, 0=will not churn")
    churn_probability: float = Field(description="Probability of churn (0.0 to 1.0)")
    model_version: Optional[str] = Field(default=None, description="Model version used for prediction")


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    customers: list[CustomerInput] = Field(description="List of customer records to predict")


class BatchPredictionOutput(BaseModel):
    """Response schema for batch predictions."""
    predictions: list[PredictionOutput]
    count: int = Field(description="Number of predictions made")


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""
    status: str = Field(description="Service status", examples=["healthy", "degraded"])
    model_loaded: bool = Field(description="Whether a model is currently loaded")
    model_version: Optional[str] = Field(default=None, description="Loaded model version")
