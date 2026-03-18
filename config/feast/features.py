"""
Feast feature definitions for the Telco Churn project.

We organize features into three FeatureViews, each representing a logical
grouping of customer attributes:
1. customer_demographics — age, gender, partner status
2. customer_account — tenure, contract, billing, charges
3. customer_services — phone, internet, add-on services

Each FeatureView reads from a Parquet file (FileSource) and uses
customerID as the entity key.
"""
import os
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# Resolve data path relative to this file so it works locally and in Docker
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "processed", "train.parquet"))


# === Entity ===
# Every feature is associated with a customer
customer = Entity(
    name="customer",
    join_keys=["customerID"],
    description="Unique customer identifier",
)

# === File Sources ===
# All point to the processed training data
customer_source = FileSource(
    name="customer_source",
    path=_DATA_PATH,
    timestamp_field="event_timestamp",
)

# === Feature Views ===

customer_demographics = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="gender", dtype=String),
        Field(name="SeniorCitizen", dtype=Int32),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
    ],
    source=customer_source,
    description="Customer demographic features",
)

customer_account = FeatureView(
    name="customer_account",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure", dtype=Int32),
        Field(name="Contract", dtype=String),
        Field(name="PaperlessBilling", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
    source=customer_source,
    description="Customer account and billing features",
)

customer_services = FeatureView(
    name="customer_services",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="PhoneService", dtype=String),
        Field(name="MultipleLines", dtype=String),
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="OnlineBackup", dtype=String),
        Field(name="DeviceProtection", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
    ],
    source=customer_source,
    description="Customer service subscription features",
)
