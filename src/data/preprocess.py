"""
Feature engineering for the Telco Churn dataset.

Creates derived features that improve model performance:
- Tenure buckets (categorical grouping)
- Service count (total number of services subscribed)
- Monthly charges per tenure month
- Automatic payment flag
- One-hot encoding of categorical variables
"""
import pandas as pd
import numpy as np


# Columns that indicate internet-based services
INTERNET_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# All categorical columns that need encoding
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Numeric columns used as features
NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
]


def create_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Group tenure into meaningful buckets.

    Buckets:
    - 0-12 months: New customers (highest churn risk)
    - 13-24 months: Short-term
    - 25-48 months: Medium-term
    - 49-60 months: Long-term
    - 61+ months: Loyal customers (lowest churn risk)
    """
    df = df.copy()
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    return df


def count_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count total number of services each customer subscribes to.

    Includes PhoneService + all 6 internet-based services.
    Customers with more services tend to have lower churn.
    """
    df = df.copy()

    # Count phone service
    service_count = (df["PhoneService"] == "Yes").astype(int)

    # Count internet-based services
    for col in INTERNET_SERVICES:
        service_count += (df[col] == "Yes").astype(int)

    df["service_count"] = service_count
    return df


def compute_charges_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived charge-related features.

    - avg_monthly_charge: TotalCharges / tenure (avg spend per month)
      For tenure=0, we use MonthlyCharges as the value.
    """
    df = df.copy()
    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    return df


def flag_automatic_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers using automatic payment methods.

    Automatic payment customers tend to have lower churn rates.
    """
    df = df.copy()
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns.

    Uses drop_first=True to avoid multicollinearity.
    Also encodes the tenure_bucket if present.
    """
    df = df.copy()

    encode_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if "tenure_bucket" in df.columns:
        encode_cols.append("tenure_bucket")

    df = pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (everything except target and IDs)."""
    exclude = {"customerID", "Churn", "event_timestamp"}
    return [c for c in df.columns if c not in exclude]


def preprocess_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps: tenure buckets → service count → charge features → auto payment → encoding
    """
    df = create_tenure_buckets(df)
    df = count_services(df)
    df = compute_charges_features(df)
    df = flag_automatic_payment(df)
    df = encode_categoricals(df)
    return df


def preprocess_for_serving(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a single customer record for prediction.

    Same pipeline as training, but designed to work with 1+ rows.
    """
    return preprocess_for_training(df)
