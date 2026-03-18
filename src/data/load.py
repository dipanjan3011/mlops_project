"""
Data loading and splitting utilities for the Telco Churn dataset.

This module handles:
- Loading the raw CSV dataset
- Basic cleaning (TotalCharges whitespace issue)
- Train/test splitting with stratification
- Saving processed data as Parquet files
"""
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


# Path configuration — works both locally and inside Docker
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Churn CSV file.

    Returns the raw DataFrame as-is from the CSV, without any transformations.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quality issues in the raw dataset.

    Issues handled:
    - TotalCharges has 11 rows with whitespace " " instead of numbers (tenure=0 customers).
      We convert to numeric, which turns these into NaN, then fill with 0.0.
    - SeniorCitizen is 0/1 int — we keep it numeric (no conversion needed).
    """
    df = df.copy()

    # Fix TotalCharges: whitespace → NaN → 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    return df


def add_event_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add event_timestamp column required by Feast.

    Feast requires an event_timestamp for point-in-time joins.
    We use the current timestamp for all rows since this is a static dataset.
    """
    df = df.copy()
    df["event_timestamp"] = pd.Timestamp(datetime.now(), tz="UTC")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn column from Yes/No to 1/0."""
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets with stratification on Churn.

    Stratification ensures both sets maintain the ~26.5% churn rate.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["Churn"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = PROCESSED_DIR):
    """Save processed train/test DataFrames as Parquet files.

    Also saves the training set as reference.parquet for Evidently drift detection.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    # Save reference data for Evidently drift comparison
    train_df.to_parquet(os.path.join(output_dir, "reference.parquet"), index=False)

    print(f"Saved processed data to {output_dir}/")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    print(f"  Reference: {len(train_df)} rows (copy of train for drift detection)")


if __name__ == "__main__":
    # Run the full data pipeline
    print("Loading raw data...")
    df = load_raw_data()
    print(f"  Raw shape: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)

    print("Adding event timestamps...")
    df = add_event_timestamp(df)

    print("Encoding target variable...")
    df = encode_target(df)

    print("Splitting data...")
    train_df, test_df = split_data(df)

    print("Saving processed data...")
    save_processed(train_df, test_df)

    print("Done!")
