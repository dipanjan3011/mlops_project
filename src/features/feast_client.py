"""
Feast client wrapper for the Telco Churn project.

Provides high-level functions to interact with the Feast feature store:
- apply(): Register feature definitions with the store
- materialize(): Push feature values from offline to online store (Redis)
- get_training_features(): Retrieve historical features for model training
- get_online_features(): Retrieve real-time features for serving
"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from feast import FeatureStore, RepoConfig
from feast.repo_config import RegistryConfig


# Feast repo path — where feature_store.yaml lives
FEAST_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "feast"
)

# All feature references across the 3 feature views
FEATURE_REFS = [
    "customer_demographics:gender",
    "customer_demographics:SeniorCitizen",
    "customer_demographics:Partner",
    "customer_demographics:Dependents",
    "customer_account:tenure",
    "customer_account:Contract",
    "customer_account:PaperlessBilling",
    "customer_account:PaymentMethod",
    "customer_account:MonthlyCharges",
    "customer_account:TotalCharges",
    "customer_services:PhoneService",
    "customer_services:MultipleLines",
    "customer_services:InternetService",
    "customer_services:OnlineSecurity",
    "customer_services:OnlineBackup",
    "customer_services:DeviceProtection",
    "customer_services:TechSupport",
    "customer_services:StreamingTV",
    "customer_services:StreamingMovies",
]


def get_store() -> FeatureStore:
    """Create a Feast FeatureStore client.

    Respects REDIS_HOST/REDIS_PORT env vars so it works both locally
    (localhost) and inside Docker (redis service name).
    """
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")
    registry_path = os.path.join(FEAST_REPO_PATH, "data", "registry.db")

    config = RepoConfig(
        project="telco_churn",
        provider="local",
        registry=registry_path,
        online_store={
            "type": "redis",
            "connection_string": f"{redis_host}:{redis_port}",
        },
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
        repo_path=FEAST_REPO_PATH,
    )
    return FeatureStore(config=config)


def apply():
    """Register feature definitions with the Feast registry.

    This reads the feature_store.yaml and features.py to set up
    entities, feature views, and data sources in the registry.
    Must be run before materialization or feature retrieval.
    """
    store = get_store()
    store.apply([])  # Feast auto-discovers from the repo
    print("Feast feature definitions applied successfully.")


def materialize(start_date: str = None, end_date: str = None):
    """Materialize features from offline store (Parquet) to online store (Redis).

    This pushes the latest feature values into Redis so they can be
    retrieved with low latency during model serving.

    Args:
        start_date: ISO format start date (default: 1 year ago)
        end_date: ISO format end date (default: now)
    """
    store = get_store()

    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.fromisoformat(end_date)

    if start_date is None:
        start_dt = end_dt - timedelta(days=365)
    else:
        start_dt = datetime.fromisoformat(start_date)

    store.materialize(start_date=start_dt, end_date=end_dt)
    print(f"Features materialized from {start_dt} to {end_dt}")


def get_training_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve historical features for model training.

    Uses point-in-time joins to get the correct feature values
    as of each entity's event_timestamp.

    Args:
        entity_df: DataFrame with 'customerID' and 'event_timestamp' columns

    Returns:
        DataFrame with all features joined to the entity DataFrame
    """
    store = get_store()

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    return training_df


def get_online_features(customer_ids: list) -> dict:
    """Retrieve real-time features from the online store (Redis).

    Used during model serving for low-latency feature lookup.

    Args:
        customer_ids: List of customer IDs to look up

    Returns:
        Dict of feature name → list of values
    """
    store = get_store()

    entity_rows = [{"customerID": cid} for cid in customer_ids]

    response = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=entity_rows,
    )

    return response.to_dict()


if __name__ == "__main__":
    # CLI interface for Makefile targets
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "apply":
            apply()
        elif command == "materialize":
            start = sys.argv[2] if len(sys.argv) > 2 else None
            end = sys.argv[3] if len(sys.argv) > 3 else None
            materialize(start, end)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m features.feast_client [apply|materialize]")
    else:
        print("Usage: python -m features.feast_client [apply|materialize]")
