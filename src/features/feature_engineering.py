"""
Standalone feature engineering for model serving.

This module provides feature computation that can run independently
of the Feast feature store — used as a fallback when Redis is unavailable
or when features need to be computed on-the-fly from raw input.
"""
import pandas as pd
import numpy as np


def compute_features(input_data: dict) -> pd.DataFrame:
    """Compute all features from raw customer input.

    Takes a dict of raw customer attributes (as received by the API)
    and returns a DataFrame with all engineered features ready for prediction.

    This mirrors the preprocessing pipeline but works on single records.

    Args:
        input_data: Dict with raw customer fields

    Returns:
        DataFrame with one row of engineered features
    """
    df = pd.DataFrame([input_data])

    # Compute derived features
    df = _add_tenure_bucket(df)
    df = _add_service_count(df)
    df = _add_charge_features(df)
    df = _add_auto_payment(df)
    df = _encode_categoricals(df)

    return df


def _add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add tenure bucket category."""
    df = df.copy()
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    df["tenure_bucket"] = pd.cut(
        df["tenure"].clip(0, 72), bins=bins, labels=labels, include_lowest=True
    )
    return df


def _add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count subscribed services."""
    df = df.copy()
    internet_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    count = (df["PhoneService"] == "Yes").astype(int)
    for col in internet_services:
        if col in df.columns:
            count += (df[col] == "Yes").astype(int)
    df["service_count"] = count
    return df


def _add_charge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average monthly charge."""
    df = df.copy()
    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    return df


def _add_auto_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Flag automatic payment methods."""
    df = df.copy()
    df["auto_payment"] = df["PaymentMethod"].str.contains(
        "automatic", case=False
    ).astype(int)
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "tenure_bucket",
    ]
    encode_cols = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=int)
    return df
