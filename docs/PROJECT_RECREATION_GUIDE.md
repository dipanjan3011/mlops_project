# Telco Customer Churn Prediction — Project Recreation Guide

This guide walks you through building the entire MLOps project from scratch. By following every step, you will have a fully working end-to-end ML system — from data ingestion to model serving, monitoring, and automated retraining — running on your machine with a single `docker compose up`.

Every step explains **what** you are doing and **why** you are doing it that way.

---

## Prerequisites

Before you begin, make sure you have the following installed:

| Tool | Version | Why |
|------|---------|-----|
| **Docker Desktop** | 4.x+ (with 4 GB+ RAM allocated) | Runs all services in containers so you don't need to install Postgres, Redis, etc. natively |
| **Python** | 3.11+ | The project's ML code and APIs are written in Python |
| **Git** | 2.x+ | Version control for your code |
| **GitHub account** | — | Required for the CI/CD pipelines (GitHub Actions) |
| **A code editor** | VS Code, PyCharm, etc. | To write and review code |

---

## What You Will Learn

By the end of this guide you will understand and have a working implementation of:

- **Docker & Docker Compose** — containerizing and orchestrating multi-service applications
- **MLflow** — experiment tracking, model registry, and the champion/challenger pattern
- **Apache Airflow & DAGs** — workflow orchestration for ML pipelines
- **Feast** — a feature store with offline (file) and online (Redis) stores
- **FastAPI** — building a REST API for model serving
- **Evidently** — detecting data drift in production
- **Great Expectations** — automated data validation
- **Prometheus & Grafana** — metrics collection and visualization
- **GitHub Actions** — CI (Continuous Integration), CD (Continuous Deployment), and CT (Continuous Training)
- **DVC** — data versioning
- **XGBoost** — gradient-boosted tree model for classification

---

## Part 1: Key Concepts (Quick Reference)

Before diving in, here is a brief explanation of every major concept used in this project. Each concept is also explained inline when it first appears, so you can skip this section and come back to it as a reference.

### Docker
Docker packages your application and all its dependencies into a **container** — a lightweight, isolated environment that runs the same way on any machine. Think of it as a shipping container for software: same shape everywhere, contents stay intact. Without Docker, you would need to install PostgreSQL, Redis, Airflow, MLflow, etc. directly on your machine and deal with version conflicts.

### Docker Compose
Docker Compose lets you define and run **multiple containers** from a single YAML file. Instead of starting 9 services manually with 9 separate `docker run` commands, you write one `docker-compose.yml` and run `docker compose up`. It also handles networking between containers (e.g., the API container can reach the MLflow container by its service name `mlflow`).

### MLflow
An open-source platform for managing the **ML lifecycle**. It does three things: (1) **Experiment tracking** — logs parameters, metrics, and artifacts (model files) for every training run so you can compare them later. (2) **Model Registry** — a central catalog of trained models with versions and aliases. (3) **Model serving** — can serve models as REST APIs (we use FastAPI instead for more control).

### Apache Airflow
A **workflow orchestration** platform. You define complex data/ML pipelines as Python code (called DAGs), and Airflow handles scheduling, retries, dependency ordering, and monitoring. It has a web UI where you can view DAG runs, trigger them manually, and inspect logs.

### DAG (Directed Acyclic Graph)
In Airflow, a DAG defines a workflow where tasks flow in one direction with no loops. For example: `validate data → preprocess → train → evaluate`. Each arrow is a dependency — "preprocess" won't start until "validate data" succeeds. The "acyclic" part means you can't create circular dependencies.

### Feast (Feature Store)
A system that manages **ML features** (the input variables your model uses). It stores features in two places: an **offline store** (for training — backed by Parquet files) and an **online store** (for real-time serving — backed by Redis for sub-millisecond reads). This ensures training and serving use the same feature values, preventing "training-serving skew."

### Redis
An **in-memory key-value database**. Extremely fast reads (sub-millisecond). Used here as Feast's online store so the prediction API can look up customer features instantly without reading from disk.

### CI (Continuous Integration)
The practice of **automatically running tests, linting, and security checks** every time someone pushes code or opens a pull request. Catches bugs before they reach production. In this project, GitHub Actions runs unit tests, integration tests, and security scans on every push.

### CD (Continuous Deployment)
**Automatically building and deploying** your application when code is merged to the main branch. No manual "deploy" step — push to main and the system builds Docker images, pushes them to a registry, and restarts the services.

### CT (Continuous Training)
The practice of **automatically retraining your model** when conditions change. Unlike CI/CD which deploys *code* changes, CT deploys *model* changes. In this project, CT runs weekly: it checks for data drift, retrains if drift is detected, validates the new model against the current champion, and deploys if it's better.

### GitHub Actions
GitHub's built-in **CI/CD platform**. You define workflows in YAML files under `.github/workflows/`. Each workflow has triggers (push, PR, schedule) and jobs (sequences of steps). Free for public repositories.

### Prometheus
A **time-series database** that scrapes metrics from your services at regular intervals. It collects numbers like "how many predictions were made," "what was the average latency," and "is data drift detected." Think of it as a data collector that watches your application's vital signs.

### Grafana
A **visualization platform** that reads data from Prometheus and displays dashboards with charts and gauges. It's the "cockpit display" for your ML system — you can see API latency, prediction distribution, model metrics, and data drift scores at a glance.

### Evidently
An open-source tool for **ML monitoring and data drift detection**. It compares your production data against a reference dataset (usually training data) to detect when the data distribution has shifted. If customers suddenly start behaving differently than they did when you trained the model, Evidently flags it.

### Great Expectations
A **data validation** framework. It lets you define "expectations" about your data (e.g., "column tenure must be between 0 and 72," "Churn must be Yes or No") and automatically checks them. Like unit tests, but for data. Prevents "garbage in, garbage out."

### DVC (Data Version Control)
**"Git for data."** Git is designed for code (small text files), not large data files. DVC tracks datasets separately — your Git repo stores a small pointer file, and DVC manages the actual data. This lets you reproduce any previous version of your data pipeline.

### FastAPI
A modern Python **web framework** for building REST APIs. It auto-generates interactive documentation (Swagger UI at `/docs`), validates request/response data using Pydantic schemas, and runs asynchronously for high throughput.

### XGBoost
A **gradient-boosted decision tree** library. It's one of the most popular ML algorithms for structured/tabular data. It works by training many small decision trees sequentially, where each tree corrects the mistakes of the previous ones. Fast, accurate, and handles missing values well.

---

## Part 2: Project Foundation

### Step 2.1: Create the Project Skeleton

Create the directory structure. Each directory has a specific purpose:

```bash
mkdir -p mlops_project
cd mlops_project

# Source code — organized by concern
mkdir -p src/data          # Data loading and preprocessing
mkdir -p src/features      # Feature store client and engineering
mkdir -p src/models        # Training, prediction, evaluation
mkdir -p src/monitoring    # Drift detection and metrics
mkdir -p src/serving       # FastAPI application
mkdir -p src/validation    # Data validation with Great Expectations

# Configuration — one folder per service
mkdir -p config/feast                              # Feature store config
mkdir -p config/grafana/provisioning/datasources   # Grafana auto-config
mkdir -p config/grafana/provisioning/dashboards    # Grafana dashboard provisioning
mkdir -p config/grafana/dashboards                 # Dashboard JSON files
mkdir -p config/prometheus                         # Prometheus scrape config
mkdir -p config/evidently                          # Drift detection config

# Airflow DAGs — Airflow discovers DAGs from this directory
mkdir -p dags

# Data — raw input and processed output
mkdir -p data/raw data/processed data/external

# Docker — one Dockerfile per service
mkdir -p docker

# Tests — separated into fast (unit) and slow (integration)
mkdir -p tests/unit tests/integration

# CI/CD — GitHub Actions workflows
mkdir -p .github/workflows

# Documentation
mkdir -p docs
```

**Why this structure?** Separation of concerns. Source code lives in `src/` (organized by what it does, not what tool it uses). Infrastructure config lives in `config/`. Orchestration logic lives in `dags/`. Containerization files live in `docker/`. This makes it easy for anyone to find what they're looking for.

### Step 2.2: Initialize Git

```bash
git init
```

Create a `.gitignore` file to tell Git which files to skip. This is important because you don't want to commit large data files, secrets, or generated artifacts.

Create the file `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
*.egg
dist/
build/
.eggs/
*.whl

# Virtual environments
.venv/
venv/
env/
ENV/

# Docker
docker-compose.override.yml

# Data — large files managed by DVC, not Git
data/raw/*
data/processed/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep
*.csv
*.parquet
*.h5
*.hdf5
*.pkl
*.pickle

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.project
.settings/

# MLflow
mlruns/
mlflow/artifacts/
mlartifacts/

# DVC
/storage/
*.dvc.lock

# Airflow
airflow.db
airflow.cfg
webserver_config.py
logs/
airflow-worker.pid
standalone_admin_password.txt

# Feast
feature_repo/data/
config/feast/data/registry.db
.feast/

# Notebooks
.ipynb_checkpoints/
notebooks/drift_report.html

# Environment — NEVER commit .env (contains passwords)
.env

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
.ruff_cache/

# Misc
*.log
*.bak
*.tmp
```

**Why each section matters:**
- **Data files** (`.csv`, `.parquet`) are excluded because they're large and managed by DVC instead of Git.
- **`.env`** is excluded because it contains passwords and secrets. You commit `.env.example` (with placeholder values) so people know what variables to set.
- **`__pycache__/`** is excluded because Python generates these bytecode files automatically — they're machine-specific and not source code.
- **`mlruns/`** is excluded because MLflow artifacts can be very large (model files) and are managed by the MLflow server.

Create `.gitkeep` files so Git tracks the empty data directories:

```bash
touch data/raw/.gitkeep data/processed/.gitkeep data/external/.gitkeep
```

### Step 2.3: Create the Python Package

Create `setup.py` — this makes your `src/` directory an installable Python package:

```python
from setuptools import setup, find_packages

setup(
    name="mlops_project",
    version="0.1.0",
    description="Telco Customer Churn Prediction - MLOps Demo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
```

**Why `package_dir={"": "src"}`?** This tells Python that packages live inside the `src/` directory. So `src/models/train.py` can be imported as `from models.train import train` instead of `from src.models.train import train`. Cleaner imports.

Create `requirements.txt` — all dependencies with **pinned versions**:

```txt
# ML Libraries
scikit-learn==1.3.2
xgboost==2.0.3
pandas==2.1.4
numpy==1.26.2

# Feature Store
feast==0.40.1
redis==5.0.1

# Experiment Tracking
mlflow==2.9.2
psycopg2-binary==2.9.9

# Orchestration — installed in Docker only via Dockerfile.airflow
# apache-airflow==2.8.0

# Model Serving
fastapi==0.108.0
uvicorn==0.25.0
pydantic==2.5.3

# Monitoring
evidently==0.5.0
prometheus-client==0.19.0

# Data Validation
great-expectations==0.18.8

# Data Versioning
dvc==3.38.1

# Testing
pytest==7.4.4
httpx==0.26.0

# Utilities
python-dotenv==1.0.0
PyYAML==6.0.1
joblib==1.3.2
pyarrow==14.0.2
```

**Why pin versions?** Reproducibility. If you install `pandas` without a version, pip might install a newer version that changes behavior or breaks compatibility. Pinning ensures everyone gets the exact same environment.

**Why is Airflow commented out?** Airflow has many dependencies that can conflict with other packages. It's installed only inside its Docker container via a dedicated Dockerfile, keeping the local environment clean.

### Step 2.4: Create the Environment File

Create `.env.example` — a template showing all required environment variables:

```env
# PostgreSQL (shared by MLflow + Airflow)
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops_password
POSTGRES_DB=mlops

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlops:mlops_password@postgres:5432/mlops
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://mlops:mlops_password@postgres:5432/mlops
AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
AIRFLOW_UID=50000

# Redis (Feast online store)
REDIS_HOST=redis
REDIS_PORT=6379

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000
MODEL_NAME=churn-model
MODEL_ALIAS=champion

# Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/mlops-overview.json
```

**Why separate databases for MLflow and Airflow?** Both tools use database migrations (via Alembic). If they share the same database, their migration tables (`alembic_version`) collide. Giving each its own database prevents this conflict.

### Step 2.5: Set Up the Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Why a virtual environment?** Isolation. Without it, packages install into your system Python and can conflict with other projects. The `-e .` flag installs your project in "editable" mode — changes to `src/` files take effect immediately without reinstalling.

Create `__init__.py` files in every `src/` subdirectory to make them proper Python packages:

```bash
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/monitoring/__init__.py
touch src/serving/__init__.py
touch src/validation/__init__.py
```

### Checkpoint: Verify Foundation

```bash
python -c "import models; print('Package import works!')"
```

---

## Part 3: The Data Layer

### Step 3.1: Get the Dataset

Download the **Telco Customer Churn** dataset from Kaggle:
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download the CSV file
3. Place it at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains **7,043 customers** with 21 columns including demographics, account info, services subscribed, and whether they churned. The churn rate is about **26.5%** (imbalanced dataset — most customers don't churn).

### Step 3.2: Data Loading Module

Create `src/data/load.py`:

```python
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
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
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


def save_processed(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = PROCESSED_DIR
):
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
```

**Why each step:**
- **`clean_data`**: The raw dataset has a known issue — 11 rows have whitespace `" "` in the TotalCharges column instead of a number. These are customers with `tenure=0` (brand new). We convert to numeric and fill NaN with 0.0.
- **`add_event_timestamp`**: Feast requires a timestamp column for "point-in-time joins" — this prevents data leakage by ensuring you only use feature values that were available at the time of the event.
- **`encode_target`**: ML models need numeric inputs. We convert `"Yes"/"No"` to `1/0`.
- **`split_data` with stratification**: When the dataset is imbalanced (26.5% churn), random splitting might give you a test set with very different churn rate. Stratification ensures both train and test maintain the same ratio.
- **`reference.parquet`**: A snapshot of the training data saved separately. Later, Evidently compares new data against this reference to detect drift.

### Step 3.3: Preprocessing Module

Create `src/data/preprocess.py`:

```python
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
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# All categorical columns that need encoding
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numeric columns used as features
NUMERIC_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
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
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )
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
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    return df


def flag_automatic_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers using automatic payment methods.

    Automatic payment customers tend to have lower churn rates.
    """
    df = df.copy()
    df["auto_payment"] = (
        df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    )
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
```

**Why feature engineering?**
- **Tenure buckets**: A customer with 1 month tenure behaves very differently from one with 60 months. Buckets capture these non-linear patterns.
- **Service count**: Customers who subscribe to more services are less likely to churn (they're more invested). This single number captures the pattern better than 7 separate binary columns.
- **avg_monthly_charge**: Normalizes spending by tenure length to compare customers fairly.
- **auto_payment flag**: Customers on auto-pay are less likely to churn (less friction to stay).
- **One-hot encoding with `drop_first=True`**: XGBoost needs numeric inputs. We convert categorical strings to binary columns. `drop_first` avoids perfect multicollinearity (e.g., if gender has two values, you only need one column).

### Step 3.4: Data Validation

Create `src/validation/validate.py`:

```python
"""
Data validation using Great Expectations.

Defines two validation suites:
1. Raw data suite — validates the CSV file before processing
2. Processed data suite — validates engineered features after preprocessing

These validations run as part of the Airflow training pipeline to catch
data quality issues before they affect model training.
"""

import os
import pandas as pd
import great_expectations as gx


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def validate_raw_data(df: pd.DataFrame) -> dict:
    """Validate the raw Telco Churn dataset.

    Checks:
    - Expected number of columns (21)
    - No null customerIDs
    - Churn values are only Yes/No
    - tenure is between 0 and 72
    - MonthlyCharges is positive
    - SeniorCitizen is only 0 or 1
    - Expected row count range (6000-8000 for this dataset)

    Returns dict with 'success' boolean and 'results' details.
    """
    context = gx.get_context()

    datasource = context.sources.add_or_update_pandas(name="raw_data")
    data_asset = datasource.add_dataframe_asset(name="raw_telco")
    batch_request = data_asset.build_batch_request(dataframe=df)

    # Create expectation suite
    context.add_or_update_expectation_suite("raw_data_suite")

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="raw_data_suite",
    )

    # Column count check
    validator.expect_table_column_count_to_equal(21)

    # Row count sanity check
    validator.expect_table_row_count_to_be_between(min_value=6000, max_value=8000)

    # customerID should never be null
    validator.expect_column_values_to_not_be_null("customerID")

    # Churn must be Yes or No
    validator.expect_column_values_to_be_in_set("Churn", ["Yes", "No"])

    # tenure should be 0-72
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=72)

    # MonthlyCharges should be positive
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    # SeniorCitizen is binary
    validator.expect_column_values_to_be_in_set("SeniorCitizen", [0, 1])

    # Validate
    results = validator.validate()

    return {
        "success": results.success,
        "statistics": results.statistics,
        "suite": "raw_data_suite",
    }


def validate_processed_data(df: pd.DataFrame) -> dict:
    """Validate the processed/engineered dataset.

    Checks:
    - Churn is encoded as 0/1
    - TotalCharges is numeric (no whitespace strings)
    - No null values in key columns
    - service_count is between 0 and 7
    - Feature columns exist

    Returns dict with 'success' boolean and 'results' details.
    """
    context = gx.get_context()

    datasource = context.sources.add_or_update_pandas(name="processed_data")
    data_asset = datasource.add_dataframe_asset(name="processed_telco")
    batch_request = data_asset.build_batch_request(dataframe=df)

    context.add_or_update_expectation_suite("processed_data_suite")

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="processed_data_suite",
    )

    # Churn should now be 0/1
    validator.expect_column_values_to_be_in_set("Churn", [0, 1])

    # TotalCharges should be numeric with no nulls
    validator.expect_column_values_to_not_be_null("TotalCharges")
    validator.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # service_count should be 0-7
    if "service_count" in df.columns:
        validator.expect_column_values_to_be_between(
            "service_count", min_value=0, max_value=7
        )

    # No nulls in critical columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]:
        validator.expect_column_values_to_not_be_null(col)

    results = validator.validate()

    return {
        "success": results.success,
        "statistics": results.statistics,
        "suite": "processed_data_suite",
    }


if __name__ == "__main__":
    from data.load import load_raw_data, clean_data, encode_target
    from data.preprocess import preprocess_for_training

    print("Validating raw data...")
    raw_df = load_raw_data()
    raw_result = validate_raw_data(raw_df)
    print(f"  Raw validation: {'PASSED' if raw_result['success'] else 'FAILED'}")
    print(f"  Stats: {raw_result['statistics']}")

    print("\nValidating processed data...")
    processed_df = clean_data(raw_df)
    processed_df = encode_target(processed_df)
    processed_df = preprocess_for_training(processed_df)
    proc_result = validate_processed_data(processed_df)
    print(f"  Processed validation: {'PASSED' if proc_result['success'] else 'FAILED'}")
    print(f"  Stats: {proc_result['statistics']}")
```

**Why validate data before training?** If the data file gets corrupted, a column gets renamed, or someone uploads the wrong file, validation catches it *before* you waste time training a model on garbage data. The validation suite acts as a contract: "this data must have exactly 21 columns, tenure between 0-72, etc."

### Step 3.5: Data Versioning with DVC

Create `dvc.yaml`:

```yaml
stages:
  preprocess:
    cmd: python -m data.load
    deps:
      - data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
      - src/data/load.py
      - src/data/preprocess.py
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet
      - data/processed/reference.parquet

  validate:
    cmd: python -m validation.validate
    deps:
      - data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
      - data/processed/train.parquet
      - src/validation/validate.py
```

**Why DVC?** Datasets change over time — new data arrives, bugs are fixed, features are re-engineered. DVC tracks which version of the data produced which model. You can reproduce any previous pipeline run by checking out the corresponding Git commit.

The `stages` define a pipeline: `preprocess` depends on the raw CSV and produces Parquet files. `validate` depends on those Parquet files. DVC knows the dependency chain and only reruns stages whose inputs changed.

---

## Part 4: The Model Layer

### Step 4.1: Hyperparameter Configuration

Create `src/models/hyperparams.py`:

```python
"""
Default hyperparameter configurations for model training.

The churn dataset has a class imbalance (~26.5% positive class),
so we use scale_pos_weight to compensate. This is calculated as:
  (count of negatives) / (count of positives) ≈ 73.5 / 26.5 ≈ 2.77 → rounded to 3.0
"""

# XGBoost default configuration
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "scale_pos_weight": 3.0,  # Compensate for 26.5% churn imbalance
    "eval_metric": "logloss",
    "random_state": 42,
    "use_label_encoder": False,
}

# Hyperparameter search space (for future Optuna integration)
XGBOOST_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [2.0, 3.0, 4.0],
}

# Model registry settings
MODEL_NAME = "churn-model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
EXPERIMENT_NAME = "telco-churn-experiment"
```

**Why `scale_pos_weight=3.0`?** The dataset is imbalanced: ~73.5% no-churn vs ~26.5% churn. Without compensation, the model would learn to just predict "no churn" for everything (and still be 73.5% accurate!). `scale_pos_weight` tells XGBoost to pay 3x more attention to the minority class (churners).

**Why a separate config file?** Keeps training code clean. You can change hyperparameters without touching the training logic. The search space is ready for future hyperparameter optimization.

### Step 4.2: Model Training with MLflow

Create `src/models/train.py`:

```python
"""
Model training with MLflow experiment tracking.

Trains an XGBoost classifier on the Telco Churn dataset, logs all
parameters, metrics, and artifacts to MLflow, and registers the model
in the MLflow Model Registry with a 'champion' alias.

Usage:
    python -m models.train
"""

import os
import warnings

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.load import (
    load_raw_data,
    clean_data,
    encode_target,
    add_event_timestamp,
    split_data,
    save_processed,
)
from data.preprocess import preprocess_for_training, get_feature_columns
from models.hyperparams import (
    XGBOOST_PARAMS,
    MODEL_NAME,
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
)

warnings.filterwarnings("ignore")

# MLflow connection — uses env var in Docker, defaults to localhost for local dev
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def train(tracking_uri: str = MLFLOW_TRACKING_URI):
    """Run the full training pipeline.

    Steps:
    1. Load and preprocess data
    2. Split into train/test
    3. Train XGBoost with MLflow autologging
    4. Log additional custom metrics
    5. Register model and set champion alias
    """
    # Set MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # === Data Pipeline ===
    print("Loading and preprocessing data...")
    df = load_raw_data()
    df = clean_data(df)
    df = add_event_timestamp(df)
    df = encode_target(df)

    # Split before feature engineering to prevent data leakage
    train_df, test_df = split_data(df)
    save_processed(train_df, test_df)

    # Feature engineering
    train_processed = preprocess_for_training(train_df)
    test_processed = preprocess_for_training(test_df)

    # Align columns — ensure test has same columns as train
    feature_cols = get_feature_columns(train_processed)

    # Add any missing columns to test (from one-hot encoding differences)
    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0

    X_train = train_processed[feature_cols]
    y_train = train_processed["Churn"]
    X_test = test_processed[feature_cols]
    y_test = test_processed["Churn"]

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.3f}")

    # === Training with MLflow ===
    # Enable autologging — captures params, metrics, model artifact automatically
    mlflow.xgboost.autolog(log_models=True)

    with mlflow.start_run(run_name="xgboost-churn") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        # Train the model
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_test, y_test)
            ],  # Required for autolog to capture validation metrics
            verbose=False,
        )

        # === Custom Metrics ===
        # Autolog captures some metrics, but we log additional business-relevant ones
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
        }

        mlflow.log_metrics(metrics)

        # Log feature columns for serving alignment
        mlflow.log_dict({"feature_columns": feature_cols}, "feature_columns.json")

        print("\n=== Test Metrics ===")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # === Model Registry ===
        # Register the model and set it as champion
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)

        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=registered_model.version,
        )

        print(f"\nModel registered: {MODEL_NAME} v{registered_model.version}")
        print(f"Alias '{CHAMPION_ALIAS}' set to v{registered_model.version}")
        print(f"MLflow UI: {tracking_uri}")

    return model, metrics


if __name__ == "__main__":
    train()
```

**Key concepts explained:**

- **`mlflow.xgboost.autolog()`**: Automatically captures all XGBoost hyperparameters, training metrics, and the model artifact without you writing any logging code. One line does the work of dozens.

- **`mlflow.register_model()`**: Stores the trained model in MLflow's Model Registry — a catalog of all your models with version numbers. Each training run creates a new version (v1, v2, v3...).

- **`client.set_registered_model_alias("champion")`**: The **champion/challenger pattern**. Your serving endpoint loads whatever model is aliased as "champion." When you retrain, the new model is a "challenger." If it beats the champion, you move the alias. This is zero-downtime model deployment — the API keeps serving the old model until the alias changes.

- **`feature_columns.json`**: Saved as an artifact so the serving code knows exactly which columns the model expects, in which order. This prevents misalignment between training and serving.

- **Why split before feature engineering?** Prevents data leakage. If you compute statistics (like mean values) on the full dataset and then split, the test set's statistics leak into the training features.

### Step 4.3: Model Evaluation

Create `src/models/evaluate.py`:

```python
"""
Model evaluation and champion/challenger comparison.

Computes comprehensive metrics and decides whether a newly trained model
(challenger) should replace the current champion.
"""

import os

import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.hyperparams import MODEL_NAME, CHAMPION_ALIAS

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics for binary classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def compare_champion_challenger(
    challenger_metrics: dict,
    primary_metric: str = "f1",
    improvement_threshold: float = 0.01,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> dict:
    """Compare a challenger model's metrics against the current champion.

    The challenger is promoted to champion if it beats the champion's
    primary metric by at least improvement_threshold (default 1%).

    Args:
        challenger_metrics: Dict of metric_name → value
        primary_metric: The metric to use for comparison (default: f1)
        improvement_threshold: Minimum improvement required to promote
        tracking_uri: MLflow tracking server URI

    Returns:
        Dict with comparison results and promotion decision
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get the current champion's metrics
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        champion_run = client.get_run(champion_version.run_id)
        champion_metrics = {
            k.replace("test_", ""): v
            for k, v in champion_run.data.metrics.items()
            if k.startswith("test_")
        }
    except Exception:
        # No champion exists yet — challenger wins by default
        return {
            "promote": True,
            "reason": "No existing champion — challenger promoted by default",
            "challenger_metrics": challenger_metrics,
            "champion_metrics": None,
        }

    # Compare
    challenger_value = challenger_metrics.get(primary_metric, 0)
    champion_value = champion_metrics.get(primary_metric, 0)
    improvement = challenger_value - champion_value

    promote = improvement >= improvement_threshold

    return {
        "promote": promote,
        "reason": (
            f"Challenger {primary_metric}={challenger_value:.4f} vs "
            f"Champion {primary_metric}={champion_value:.4f} "
            f"(improvement={improvement:+.4f}, threshold={improvement_threshold})"
        ),
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "improvement": improvement,
    }


def promote_challenger(
    version: str,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Promote a challenger model version to champion.

    Updates the 'champion' alias to point to the new version.
    The previous champion can still be accessed by its version number.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=version,
    )
    print(f"Model v{version} promoted to '{CHAMPION_ALIAS}'")
```

**Why the 1% improvement threshold?** Prevents promoting a model that is only marginally better. A 0.1% F1 improvement could just be noise from random data splits. Requiring at least 1% improvement ensures the new model is meaningfully better.

### Step 4.4: Prediction Module

Create `src/models/predict.py`:

```python
"""
Model prediction utilities.

Loads the champion model from MLflow Model Registry and provides
prediction functions for single records and batches.
"""

import os
import time

import mlflow
import pandas as pd

from models.hyperparams import MODEL_NAME, CHAMPION_ALIAS

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


class ChurnPredictor:
    """Wrapper around the MLflow-registered churn model.

    Handles model loading with retry logic and provides prediction methods.
    """

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI):
        self.tracking_uri = tracking_uri
        self.model = None
        self.feature_columns = None
        self.model_version = None

    def load_model(self, max_retries: int = 3, retry_delay: float = 2.0):
        """Load the champion model from MLflow registry.

        Retries with exponential backoff in case MLflow server is starting up.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.tracking.MlflowClient()

        for attempt in range(max_retries):
            try:
                # Get champion model version
                version_info = client.get_model_version_by_alias(
                    MODEL_NAME, CHAMPION_ALIAS
                )
                self.model_version = version_info.version

                # Load the model
                model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
                self.model = mlflow.xgboost.load_model(model_uri)

                # Load feature columns
                try:
                    feature_info = mlflow.artifacts.load_dict(
                        f"runs:/{version_info.run_id}/feature_columns.json"
                    )
                    self.feature_columns = feature_info.get("feature_columns")
                except Exception:
                    self.feature_columns = None

                print(f"Loaded model: {MODEL_NAME} v{self.model_version}")
                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = retry_delay * (2**attempt)
                    print(
                        f"Model load attempt {attempt + 1} failed: {e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    print(f"Failed to load model after {max_retries} attempts: {e}")
                    return False

    def predict(self, features: pd.DataFrame) -> dict:
        """Make a prediction for a single customer.

        Args:
            features: DataFrame with one row of engineered features

        Returns:
            Dict with churn prediction and probability
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Align features with training columns
        features = self._align_features(features)

        prediction = int(self.model.predict(features)[0])
        probability = float(self.model.predict_proba(features)[0][1])

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "model_version": self.model_version,
        }

    def predict_batch(self, features: pd.DataFrame) -> list:
        """Make predictions for multiple customers.

        Args:
            features: DataFrame with multiple rows of engineered features

        Returns:
            List of prediction dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        features = self._align_features(features)

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        return [
            {
                "churn_prediction": int(pred),
                "churn_probability": round(float(prob), 4),
                "model_version": self.model_version,
            }
            for pred, prob in zip(predictions, probabilities)
        ]

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure prediction features match training feature columns.

        Adds missing columns (as 0) and removes extra columns to match
        the exact feature set the model was trained on.
        """
        if self.feature_columns is None:
            return features

        # Add missing columns
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0

        # Select only training columns in the right order
        return features[self.feature_columns]
```

**Why retry logic with exponential backoff?** When Docker Compose starts all services simultaneously, the API container might try to load the model before MLflow is fully ready. Instead of crashing, it waits (2s, 4s, 8s...) and retries.

**Why `_align_features`?** One-hot encoding can produce different columns depending on the data (e.g., a test record might not have every payment method represented). This method ensures prediction features match exactly what the model was trained on.

---

## Part 5: The Feature Store (Feast)

### Step 5.1: Feast Configuration

Create `config/feast/feature_store.yaml`:

```yaml
project: telco_churn
provider: local
registry: data/registry.db
online_store:
  type: redis
  connection_string: redis:6379
offline_store:
  type: file
entity_key_serialization_version: 3
```

**Why two stores?**
- **Offline store (file)**: Parquet files on disk. Used during training for batch retrieval of historical features. Slow but can handle large datasets.
- **Online store (Redis)**: In-memory database. Used during serving for real-time feature lookup. Sub-millisecond latency but stores only the latest feature values.

The process of copying data from offline to online is called **materialization**.

### Step 5.2: Feature Definitions

Create `config/feast/features.py`:

```python
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
```

**Why three separate FeatureViews?** Logical grouping. Demographics change rarely, account info changes monthly, services can change at any time. Separate views allow independent TTLs and materialization schedules in a production system.

**Why `ttl=timedelta(days=365)`?** TTL (Time To Live) tells Feast how long feature values are valid. After 365 days without an update, the feature is considered stale.

### Step 5.3: Feast Client

Create `src/features/feast_client.py`:

```python
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


# Feast repo path — where feature_store.yaml lives
FEAST_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "feast",
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
```

### Step 5.4: Feature Engineering for Serving

Create `src/features/feature_engineering.py`:

```python
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
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
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
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    return df


def _add_auto_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Flag automatic payment methods."""
    df = df.copy()
    df["auto_payment"] = (
        df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    )
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    categorical_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_bucket",
    ]
    encode_cols = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=int)
    return df
```

**Why a separate serving module?** The `/predict` endpoint receives raw customer data and needs to compute features on-the-fly. This module mirrors the training pipeline but works on single records. The `/predict/feast` endpoint looks up pre-computed features from Redis instead.

---

## Part 6: The Serving Layer (FastAPI)

### Step 6.1: Request/Response Schemas

Create `src/serving/schemas.py`:

```python
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
    SeniorCitizen: int = Field(
        description="Whether the customer is a senior citizen (0 or 1)", ge=0, le=1
    )
    Partner: str = Field(
        description="Whether the customer has a partner", examples=["Yes", "No"]
    )
    Dependents: str = Field(
        description="Whether the customer has dependents", examples=["Yes", "No"]
    )
    tenure: int = Field(
        description="Number of months the customer has stayed", ge=0, le=72
    )
    PhoneService: str = Field(
        description="Whether the customer has phone service", examples=["Yes", "No"]
    )
    MultipleLines: str = Field(
        description="Whether the customer has multiple lines",
        examples=["Yes", "No", "No phone service"],
    )
    InternetService: str = Field(
        description="Customer's internet service provider",
        examples=["DSL", "Fiber optic", "No"],
    )
    OnlineSecurity: str = Field(
        description="Whether the customer has online security",
        examples=["Yes", "No", "No internet service"],
    )
    OnlineBackup: str = Field(
        description="Whether the customer has online backup",
        examples=["Yes", "No", "No internet service"],
    )
    DeviceProtection: str = Field(
        description="Whether the customer has device protection",
        examples=["Yes", "No", "No internet service"],
    )
    TechSupport: str = Field(
        description="Whether the customer has tech support",
        examples=["Yes", "No", "No internet service"],
    )
    StreamingTV: str = Field(
        description="Whether the customer has streaming TV",
        examples=["Yes", "No", "No internet service"],
    )
    StreamingMovies: str = Field(
        description="Whether the customer has streaming movies",
        examples=["Yes", "No", "No internet service"],
    )
    Contract: str = Field(
        description="The contract term",
        examples=["Month-to-month", "One year", "Two year"],
    )
    PaperlessBilling: str = Field(
        description="Whether the customer has paperless billing", examples=["Yes", "No"]
    )
    PaymentMethod: str = Field(
        description="The customer's payment method",
        examples=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
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

    customerID: str = Field(
        description="Unique customer identifier to look up in the feature store"
    )


class PredictionOutput(BaseModel):
    """Response schema for a single prediction."""

    customerID: str = Field(description="Customer identifier from the request")
    churn_prediction: int = Field(
        description="Binary prediction: 1=will churn, 0=will not churn"
    )
    churn_probability: float = Field(description="Probability of churn (0.0 to 1.0)")
    model_version: Optional[str] = Field(
        default=None, description="Model version used for prediction"
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    customers: list[CustomerInput] = Field(
        description="List of customer records to predict"
    )


class BatchPredictionOutput(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionOutput]
    count: int = Field(description="Number of predictions made")


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str = Field(description="Service status", examples=["healthy", "degraded"])
    model_loaded: bool = Field(description="Whether a model is currently loaded")
    model_version: Optional[str] = Field(
        default=None, description="Loaded model version"
    )
```

**Why Pydantic schemas?** They serve three purposes: (1) **Validate** incoming requests automatically — if someone sends `tenure: "abc"`, they get a clear error. (2) **Document** the API — FastAPI generates interactive Swagger docs at `/docs` from these schemas. (3) **Type safety** — your IDE knows exactly what fields are available.

### Step 6.2: Prometheus Middleware

Create `src/serving/middleware.py`:

```python
"""
Prometheus metrics middleware for the FastAPI serving layer.

Tracks:
- Total request count by endpoint and status code
- Request latency histogram by endpoint
- Prediction distribution (churn vs no-churn counts)
- Current model version info gauge
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Info


# === Prometheus Metrics ===

# Request metrics
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions made",
    ["prediction"],  # "churn" or "no_churn"
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Model info
MODEL_INFO = Info(
    "model",
    "Currently loaded model information",
)


def record_prediction(prediction: int, probability: float):
    """Record a prediction in Prometheus metrics."""
    label = "churn" if prediction == 1 else "no_churn"
    PREDICTION_COUNT.labels(prediction=label).inc()
    PREDICTION_PROBABILITY.observe(probability)


def update_model_info(model_name: str, model_version: str):
    """Update the model info gauge."""
    MODEL_INFO.info(
        {
            "name": model_name,
            "version": str(model_version),
        }
    )


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware that tracks request count and latency.

    Wraps every request to record:
    - Response status code
    - Request duration
    Excludes /metrics endpoint to avoid self-referential tracking.
    """
    # Skip tracking for the metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response
```

### Step 6.3: FastAPI Application

Create `src/serving/app.py`:

```python
"""
FastAPI application for serving churn predictions.

Endpoints:
- POST /predict         -- Single customer churn prediction
- POST /batch-predict   -- Batch predictions for multiple customers
- GET  /health          -- Health check with model status
- GET  /metrics         -- Prometheus metrics (scraped by Prometheus)

The app loads the champion model from MLflow on startup. If the model
is not yet available (first boot before training), it returns 503 on
prediction endpoints until a model is loaded.
"""

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from serving.schemas import (
    CustomerInput,
    CustomerIDInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
)
from serving.middleware import (
    metrics_middleware,
    record_prediction,
    update_model_info,
)
from models.predict import ChurnPredictor
from features.feature_engineering import compute_features
from features.feast_client import get_online_features


# Global predictor instance
predictor = ChurnPredictor(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup with retry logic.

    Runs in background so the API starts accepting health checks immediately,
    even if the model isn't ready yet. Prediction endpoints return 503 until loaded.
    """

    # Try to load model in background -- don't block startup
    async def load_model_background():
        """Attempt model loading with exponential backoff."""
        max_attempts = 5
        for attempt in range(max_attempts):
            success = predictor.load_model(max_retries=1)
            if success:
                update_model_info(
                    model_name=os.getenv("MODEL_NAME", "churn-model"),
                    model_version=str(predictor.model_version),
                )
                print(f"Model loaded successfully: v{predictor.model_version}")
                return
            wait = min(2 ** (attempt + 1), 30)
            print(
                f"Model not available yet (attempt {attempt + 1}/{max_attempts}). Retrying in {wait}s..."
            )
            await asyncio.sleep(wait)
        print("WARNING: Could not load model. Prediction endpoints will return 503.")

    asyncio.create_task(load_model_background())
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn using XGBoost model served via MLflow",
    version="1.0.0",
    lifespan=lifespan,
)

# Add Prometheus metrics middleware
app.middleware("http")(metrics_middleware)


@app.post("/predict", response_model=PredictionOutput)
async def predict(customer: CustomerInput):
    """Predict churn for a single customer.

    Takes raw customer attributes, computes features on-the-fly,
    and returns the churn prediction with probability.

    Returns 503 if no model is loaded yet.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later.",
        )

    try:
        # Compute features from raw input
        features = compute_features(customer.model_dump(exclude={"customerID"}))

        # Make prediction
        result = predictor.predict(features)

        # Record metrics for monitoring
        record_prediction(result["churn_prediction"], result["churn_probability"])

        return PredictionOutput(
            customerID=customer.customerID,
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            model_version=result["model_version"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/feast", response_model=PredictionOutput)
async def predict_from_feast(request: CustomerIDInput):
    """Predict churn by looking up features from the Feast online store (Redis).

    Only requires a customerID — all features are fetched from Redis.
    Features must have been materialized first via `make feast-materialize`.

    Returns 503 if no model is loaded, 404 if customer not found in Feast.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later.",
        )

    try:
        # Fetch features from Feast online store (Redis)
        feast_result = get_online_features([request.customerID])

        # Check if Feast returned actual values (None means not materialized)
        if feast_result.get("tenure", [None])[0] is None:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{request.customerID}' not found in feature store. "
                "Run 'make feast-materialize' to load features into Redis.",
            )

        # Convert Feast dict to a single-row dict for compute_features
        raw_features = {k: v[0] for k, v in feast_result.items() if k != "customerID"}

        # Compute engineered features (tenure_bucket, service_count, etc.)
        features = compute_features(raw_features)

        # Make prediction
        result = predictor.predict(features)

        # Record metrics
        record_prediction(result["churn_prediction"], result["churn_probability"])

        return PredictionOutput(
            customerID=request.customerID,
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            model_version=result["model_version"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feast prediction failed: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionOutput)
async def batch_predict(batch: BatchPredictionInput):
    """Predict churn for multiple customers in a single request.

    Returns 503 if no model is loaded yet.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later.",
        )

    try:
        predictions = []
        for customer in batch.customers:
            features = compute_features(customer.model_dump(exclude={"customerID"}))
            result = predictor.predict(features)
            record_prediction(result["churn_prediction"], result["churn_probability"])
            predictions.append(
                PredictionOutput(
                    customerID=customer.customerID,
                    churn_prediction=result["churn_prediction"],
                    churn_probability=result["churn_probability"],
                    model_version=result["model_version"],
                )
            )

        return BatchPredictionOutput(predictions=predictions, count=len(predictions))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns model loading status. Used by Docker health checks and
    load balancers to determine if the service is ready.
    """
    return HealthResponse(
        status="healthy" if predictor.model is not None else "degraded",
        model_loaded=predictor.model is not None,
        model_version=predictor.model_version,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint.

    Returns all registered Prometheus metrics in the exposition format.
    Scraped by Prometheus every 15s (configured in prometheus.yml).
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
```

**Why the lifespan pattern with background model loading?** Docker needs the API container to respond to health checks quickly. If model loading blocks startup (MLflow might take 30 seconds), Docker might kill the container thinking it's unhealthy. The background approach lets the API start immediately (returning `status: degraded` on `/health`) while loading the model asynchronously. Once loaded, it switches to `status: healthy`.

---

## Part 7: Monitoring & Drift Detection

### Step 7.1: Prometheus Metrics Definitions

Create `src/monitoring/metrics.py`:

```python
"""
ML-specific Prometheus metrics for model monitoring.

These gauges track model performance and data statistics that are
updated during prediction and drift detection. They complement the
API request metrics defined in serving/middleware.py.
"""

from prometheus_client import Gauge, Summary


# === Model Performance Gauges ===
# Updated when model is evaluated (training pipeline or scheduled eval)
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy on evaluation set",
)

MODEL_F1_SCORE = Gauge(
    "model_f1_score",
    "Current model F1 score on evaluation set",
)

MODEL_PRECISION = Gauge(
    "model_precision",
    "Current model precision on evaluation set",
)

MODEL_RECALL = Gauge(
    "model_recall",
    "Current model recall on evaluation set",
)

MODEL_ROC_AUC = Gauge(
    "model_roc_auc",
    "Current model ROC AUC on evaluation set",
)

# === Data Statistics Gauges ===
FEATURE_MEAN = Gauge(
    "feature_mean",
    "Mean value of numeric features in recent predictions",
    ["feature_name"],
)

FEATURE_STD = Gauge(
    "feature_std",
    "Standard deviation of numeric features in recent predictions",
    ["feature_name"],
)

# === Prediction Statistics ===
CHURN_RATE = Gauge(
    "predicted_churn_rate",
    "Rolling churn rate in recent predictions",
)

PREDICTION_LATENCY = Summary(
    "model_prediction_latency_seconds",
    "Time spent computing model predictions (excluding feature engineering)",
)


def update_model_metrics(metrics: dict):
    """Update all model performance gauges.

    Called after model evaluation with a dict of metric_name -> value.
    """
    metric_map = {
        "accuracy": MODEL_ACCURACY,
        "f1": MODEL_F1_SCORE,
        "precision": MODEL_PRECISION,
        "recall": MODEL_RECALL,
        "roc_auc": MODEL_ROC_AUC,
    }

    for name, gauge in metric_map.items():
        if name in metrics:
            gauge.set(metrics[name])


def update_feature_stats(feature_name: str, mean: float, std: float):
    """Update feature statistics gauges."""
    FEATURE_MEAN.labels(feature_name=feature_name).set(mean)
    FEATURE_STD.labels(feature_name=feature_name).set(std)
```

### Step 7.2: Drift Detection

Create `src/monitoring/drift_detector.py`:

```python
"""
Data and model drift detection using Evidently AI.

Compares current production data against a reference dataset (training data)
to detect distribution shifts that may degrade model performance.

Exports drift scores as Prometheus gauges for Grafana dashboards and
generates HTML reports for detailed analysis.
"""

import os
from datetime import datetime

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
)
from prometheus_client import Gauge


# === Prometheus Gauges for Drift ===
DATASET_DRIFT_SCORE = Gauge(
    "dataset_drift_share",
    "Share of drifted features in the dataset",
)

DATASET_DRIFT_DETECTED = Gauge(
    "dataset_drift_detected",
    "Whether dataset drift was detected (1=yes, 0=no)",
)

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Drift score for individual features",
    ["feature_name"],
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "reports")


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_share_threshold: float = 0.3,
) -> dict:
    """Check for data drift between reference and current datasets.

    Uses Evidently's DatasetDriftMetric to determine if the overall
    dataset has drifted, and DataDriftTable for per-feature drift scores.

    Args:
        reference_df: Training/reference dataset
        current_df: Current/production dataset
        drift_share_threshold: Fraction of features that must drift
            to declare dataset-level drift (default: 30%)

    Returns:
        Dict with drift_detected, drift_share, and per-feature scores
    """
    # Select only numeric and relevant columns for drift analysis
    exclude_cols = {"customerID", "event_timestamp", "Churn"}
    numeric_cols = [
        c
        for c in reference_df.select_dtypes(include=["number"]).columns
        if c not in exclude_cols
    ]

    ref_subset = reference_df[numeric_cols].copy()
    cur_subset = current_df[numeric_cols].copy()

    # Build Evidently report
    report = Report(
        metrics=[
            DatasetDriftMetric(drift_share=drift_share_threshold),
            DataDriftTable(),
        ]
    )

    report.run(reference_data=ref_subset, current_data=cur_subset)
    result = report.as_dict()

    # Extract results
    dataset_drift = result["metrics"][0]["result"]
    drift_table = result["metrics"][1]["result"]

    drift_detected = dataset_drift["dataset_drift"]
    drift_share = dataset_drift["share_of_drifted_columns"]

    # Update Prometheus gauges
    DATASET_DRIFT_SCORE.set(drift_share)
    DATASET_DRIFT_DETECTED.set(1 if drift_detected else 0)

    # Per-feature drift scores
    feature_scores = {}
    if "drift_by_columns" in drift_table:
        for col_name, col_data in drift_table["drift_by_columns"].items():
            score = col_data.get("drift_score", 0)
            feature_scores[col_name] = score
            FEATURE_DRIFT_SCORE.labels(feature_name=col_name).set(score)

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "feature_scores": feature_scores,
        "threshold": drift_share_threshold,
    }


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = None,
) -> str:
    """Generate a detailed HTML drift report.

    Creates a comprehensive Evidently report with data drift and
    data quality analysis. Saved as an HTML file for review.

    Args:
        reference_df: Training/reference dataset
        current_df: Current/production dataset
        output_path: Where to save the HTML report

    Returns:
        Path to the generated HTML report
    """
    if output_path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")

    # Exclude non-feature columns
    exclude_cols = {"customerID", "event_timestamp"}
    cols = [c for c in reference_df.columns if c not in exclude_cols]

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    report.run(
        reference_data=reference_df[cols],
        current_data=current_df[cols],
    )

    report.save_html(output_path)
    print(f"Drift report saved to: {output_path}")

    return output_path
```

**What is data drift?** When the data your model sees in production starts looking different from the training data. Example: if customers suddenly shift from month-to-month contracts to two-year contracts, the model's predictions may become unreliable because it learned patterns from a different distribution.

**Why the 30% threshold?** If 30% or more of features have statistically drifted, we declare dataset-level drift. This triggers retraining. The threshold balances sensitivity (catching real drift) vs. false alarms (random noise).

### Step 7.3: Drift Configuration

Create `config/evidently/drift_config.yaml`:

```yaml
# Evidently drift detection configuration
#
# Defines which columns to monitor for drift and the thresholds
# for triggering alerts/retraining.

column_mapping:
  target: Churn
  numerical_features:
    - tenure
    - MonthlyCharges
    - TotalCharges
    - SeniorCitizen
  categorical_features:
    - gender
    - Partner
    - Dependents
    - PhoneService
    - MultipleLines
    - InternetService
    - OnlineSecurity
    - OnlineBackup
    - DeviceProtection
    - TechSupport
    - StreamingTV
    - StreamingMovies
    - Contract
    - PaperlessBilling
    - PaymentMethod

thresholds:
  # Fraction of features that must drift to trigger dataset-level drift
  dataset_drift_share: 0.3
  # Per-feature p-value threshold (used by Evidently's statistical tests)
  feature_drift_threshold: 0.05

alerts:
  # If drift is detected, these actions can be triggered
  on_drift_detected:
    - trigger_retraining
    - generate_report
    - notify_slack  # placeholder — configure webhook separately
```

---

## Part 8: Docker Infrastructure

### Step 8.1: Dockerfiles

#### API Dockerfile

Create `docker/Dockerfile.api`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<81" -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .

RUN mkdir -p data/raw data/processed data/external
RUN pip install -e .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Line-by-line:**
- `FROM python:3.11-slim` — Start from a minimal Python image (not the full one, to keep the image small).
- `WORKDIR /app` — All subsequent commands run inside `/app`.
- `RUN apt-get...` — Install build tools (some Python packages need C compilation) and `curl` (for health checks).
- `COPY requirements.txt` then `RUN pip install` — Install dependencies first, so Docker caches this layer. If you change code but not dependencies, the rebuild is fast.
- `COPY src/ config/ setup.py` — Copy application code (after dependencies for caching).
- `ENV PYTHONPATH=/app/src` — So `from models.train import train` resolves correctly.
- `CMD` — Start the FastAPI server with Uvicorn.

#### MLflow Dockerfile

Create `docker/Dockerfile.mlflow`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==3.10.1 \
    psycopg2-binary==2.9.9

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlops:mlops_password@postgres:5432/mlops", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "*"]
```

**Key flags:**
- `--backend-store-uri postgresql://...` — Stores experiment metadata in PostgreSQL (not SQLite) for durability and multi-user access.
- `--serve-artifacts` — MLflow serves model artifacts directly (no need for a separate artifact server).
- `--allowed-hosts "*"` — Accept connections from any host (needed for Docker networking).

#### Airflow Dockerfile

Create `docker/Dockerfile.airflow`:

```dockerfile
FROM apache/airflow:2.8.0-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies (excluding airflow itself)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    feast==0.40.1 \
    redis==5.0.1 \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    evidently==0.5.0 \
    prometheus-client==0.19.0 \
    great-expectations==0.18.8 \
    python-dotenv==1.0.0 \
    PyYAML==6.0.1 \
    joblib==1.3.2 \
    pyarrow==14.0.2 \
    "setuptools<81" \
    "alembic<1.14"

ENV PYTHONPATH=/opt/airflow/src
```

**Why extend the official Airflow image?** Airflow is complex to install from scratch. The official image handles all the system dependencies, Airflow configuration, and user permissions. We only add the ML-specific Python packages on top.

**Why `USER root` then `USER airflow`?** The official image runs as the `airflow` user for security. We temporarily switch to `root` to install system packages (`build-essential`), then switch back.

#### Database Init Script

Create `docker/init-db.sh`:

```bash
#!/bin/bash
set -e

# Create a separate database for Airflow so its alembic migrations
# don't conflict with MLflow's migrations in the 'mlops' database.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE airflow;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO $POSTGRES_USER;
EOSQL
```

**Why a separate database?** Both MLflow and Airflow use Alembic for database migrations. If they share the same database, their `alembic_version` tables collide, causing migration errors. This script creates a dedicated `airflow` database during PostgreSQL's first startup.

Make it executable:
```bash
chmod +x docker/init-db.sh
```

### Step 8.2: Docker Compose

Create `docker-compose.yml`:

```yaml
x-airflow-common: &airflow-common
  build:
    context: .
    dockerfile: docker/Dockerfile.airflow
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://mlops:mlops_password@postgres:5432/airflow
    AIRFLOW__CORE__FERNET_KEY: "81HqDtbqAywKSOumSha3BhWNOdQ26slT6K0YaZeZyPs="
    AIRFLOW__CORE__LOAD_EXAMPLES: "False"
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG: "True"
    MLFLOW_TRACKING_URI: http://mlflow:5000
    REDIS_HOST: redis
    REDIS_PORT: 6379
    PYTHONPATH: /opt/airflow/src
  volumes:
    - ./dags:/opt/airflow/dags
    - ./src:/opt/airflow/src
    - ./data:/opt/airflow/data
    - ./config:/opt/airflow/config
    - mlflow_artifacts:/mlflow/artifacts
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy

services:
  # === Storage Services ===
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlops
      POSTGRES_PASSWORD: mlops_password
      POSTGRES_DB: mlops
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlops"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # === MLflow ===
  mlflow:
    build:
      context: .
      dockerfile: docker/Dockerfile.mlflow
    ports:
      - "5001:5000"  # Use 5001 externally — port 5000 conflicts with macOS AirPlay Receiver
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: unless-stopped

  # === Airflow ===
  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db migrate &&
        airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com || true
    restart: "no"

  airflow-webserver:
    <<: *airflow-common
    command: airflow webserver --port 8080
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    depends_on:
      airflow-init:
        condition: service_completed_successfully
      mlflow:
        condition: service_healthy
    restart: unless-stopped

  airflow-scheduler:
    <<: *airflow-common
    command: airflow scheduler
    depends_on:
      airflow-init:
        condition: service_completed_successfully
      mlflow:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "airflow jobs check --job-type SchedulerJob --hostname $(hostname)"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    restart: unless-stopped

  # === Model Serving ===
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      REDIS_HOST: redis
      REDIS_PORT: 6379
      MODEL_NAME: churn-model
      MODEL_ALIAS: champion
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
      - ./data:/app/data
      - ./config:/app/config
    depends_on:
      mlflow:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: unless-stopped

  # === Monitoring ===
  prometheus:
    image: prom/prometheus:v2.48.1
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    depends_on:
      - api
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:9090/-/healthy"]
      interval: 15s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.2.3
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH: /var/lib/grafana/dashboards/mlops-overview.json
    volumes:
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - ./config/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/api/health"]
      interval: 15s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # === Redis Browser ===
  redisinsight:
    image: redis/redisinsight:latest
    ports:
      - "5540:5540"
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_artifacts:
  prometheus_data:
  grafana_data:
```

**Key design decisions explained:**

- **YAML anchor `&airflow-common`**: The DRY (Don't Repeat Yourself) pattern. All three Airflow containers share the same image, environment, and volumes. The anchor defines it once, and `<<: *airflow-common` merges it.

- **`depends_on` with `condition: service_healthy`**: Containers don't just wait for dependencies to *start* — they wait until the health check passes. This prevents "connection refused" errors during startup.

- **`service_completed_successfully`**: The `airflow-init` container runs database migrations and creates the admin user, then exits. Other Airflow containers wait for it to complete before starting.

- **Port `5001:5000` for MLflow**: macOS uses port 5000 for AirPlay Receiver. We map the container's internal port 5000 to external port 5001 to avoid the conflict.

- **Named volumes** (`postgres_data`, `redis_data`, etc.): Data persists across container restarts. Without named volumes, stopping the containers would lose all your database data.

- **`restart: unless-stopped`**: Containers automatically restart if they crash, unless you explicitly stop them with `docker compose down`.

### Step 8.3: Production Overrides

Create `docker-compose.prod.yml`:

```yaml
# Production overrides for docker-compose.yml
# Usage: docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
#
# This overrides the local dev config with:
# - GHCR images instead of local builds
# - Production environment variables
# - Restart policies and resource limits

services:
  api:
    image: ${IMAGE_PREFIX}-api:${IMAGE_TAG:-latest}
    build: !override
    restart: always
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "1.0"

  mlflow:
    image: ${IMAGE_PREFIX}-mlflow:${IMAGE_TAG:-latest}
    build: !override
    restart: always
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "0.5"

  airflow-webserver:
    image: ${IMAGE_PREFIX}-airflow:${IMAGE_TAG:-latest}
    build: !override
    restart: always

  airflow-scheduler:
    image: ${IMAGE_PREFIX}-airflow:${IMAGE_TAG:-latest}
    build: !override
    restart: always

  airflow-init:
    image: ${IMAGE_PREFIX}-airflow:${IMAGE_TAG:-latest}
    build: !override

  postgres:
    restart: always
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlops_password}

  redis:
    restart: always
```

**Why a separate prod file?** In development, you build images locally with `docker compose up --build`. In production, you pull pre-built images from GHCR (GitHub Container Registry). The prod override replaces `build:` with `image:` and adds resource limits.

### Step 8.4: Monitoring Configuration

#### Prometheus

Create `config/prometheus/prometheus.yml`:

```yaml
# Prometheus configuration for MLOps monitoring stack
#
# Scrapes metrics from the FastAPI prediction service every 15 seconds.
# The /metrics endpoint exposes both API performance metrics and ML-specific
# metrics (drift scores, prediction distribution, model info).

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # FastAPI prediction service
  - job_name: "fastapi"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
    scrape_interval: 15s
```

**Why 15s scrape interval?** Frequent enough to catch issues quickly, but not so frequent that it overwhelms the API with requests.

#### Grafana Auto-Provisioning

Create `config/grafana/provisioning/datasources/datasource.yml`:

```yaml
# Auto-provision Prometheus as the default data source.
# Grafana reads this on startup — no manual configuration needed.
apiVersion: 1

datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

Create `config/grafana/provisioning/dashboards/dashboard.yml`:

```yaml
# Auto-provision dashboards from the /var/lib/grafana/dashboards directory.
# Dashboard JSON files placed there will be automatically loaded on startup.
apiVersion: 1

providers:
  - name: "MLOps Dashboards"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

**Why auto-provisioning?** Without it, you'd have to manually add Prometheus as a data source and import dashboards every time you recreate the Grafana container. Provisioning makes it fully automatic.

For the Grafana dashboard JSON (`config/grafana/dashboards/mlops-overview.json`), you can create one via the Grafana UI and export it, or use the one included in the project. It contains panels for API request rate, prediction distribution, model metrics, and data drift scores.

---

## Part 9: Airflow DAGs (Orchestration)

### Step 9.1: Training Pipeline DAG

Create `dags/training_pipeline.py`:

```python
"""
Training Pipeline DAG — Full model training workflow.

This DAG runs the complete ML training pipeline:
1. Validate raw data with Great Expectations
2. Preprocess and engineer features
3. Apply Feast feature definitions
4. Materialize features to Redis
5. Train XGBoost model with MLflow tracking
6. Evaluate model metrics
7. Promote to champion if performance improves

Schedule: Manual trigger (can also be triggered by continuous_training DAG)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def validate_raw_data_task(**kwargs):
    """Validate the raw dataset before processing."""
    from data.load import load_raw_data
    from validation.validate import validate_raw_data

    df = load_raw_data()
    result = validate_raw_data(df)

    if not result["success"]:
        raise ValueError(f"Raw data validation failed: {result}")

    print(f"Raw data validation passed: {result['statistics']}")
    return result


def preprocess_data_task(**kwargs):
    """Load, clean, and split the data."""
    from data.load import (
        load_raw_data,
        clean_data,
        encode_target,
        add_event_timestamp,
        split_data,
        save_processed,
    )

    df = load_raw_data()
    df = clean_data(df)
    df = add_event_timestamp(df)
    df = encode_target(df)
    train_df, test_df = split_data(df)
    save_processed(train_df, test_df)

    print(f"Data preprocessed: train={len(train_df)}, test={len(test_df)}")


def feast_apply_task(**kwargs):
    """Register feature definitions with Feast."""
    from features.feast_client import apply

    apply()


def feast_materialize_task(**kwargs):
    """Push features from offline store to Redis online store."""
    from features.feast_client import materialize

    materialize()


def train_model_task(**kwargs):
    """Train XGBoost model with MLflow tracking."""
    import os
    from models.train import train

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model, metrics = train(tracking_uri=tracking_uri)

    # Push metrics to XCom for the evaluation task
    kwargs["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


def evaluate_model_task(**kwargs):
    """Evaluate trained model and decide on promotion."""
    import os
    from models.evaluate import compare_champion_challenger

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="train_model", key="metrics")

    if metrics is None:
        raise ValueError("No metrics received from training task")

    # Remove 'test_' prefix for comparison
    clean_metrics = {k.replace("test_", ""): v for k, v in metrics.items()}

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    comparison = compare_champion_challenger(
        challenger_metrics=clean_metrics,
        tracking_uri=tracking_uri,
    )

    print(f"Evaluation result: {comparison['reason']}")
    print(f"Promote: {comparison['promote']}")

    return comparison


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Full ML model training pipeline: validate → preprocess → train → evaluate",
    schedule=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training"],
) as dag:
    validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data_task,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data_task,
    )

    feast_apply = PythonOperator(
        task_id="feast_apply",
        python_callable=feast_apply_task,
    )

    feast_materialize = PythonOperator(
        task_id="feast_materialize",
        python_callable=feast_materialize_task,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task,
    )

    done = EmptyOperator(task_id="done")

    # Pipeline: validate → preprocess → feast → train → evaluate → done
    (
        validate
        >> preprocess
        >> feast_apply
        >> feast_materialize
        >> train_model
        >> evaluate
        >> done
    )
```

**Why `schedule=None`?** This DAG should only run when explicitly triggered — either manually from the Airflow UI or by the `continuous_training` DAG. You don't want accidental retraining on a schedule.

**Why XCom for passing metrics?** Airflow tasks run in separate processes. XCom (cross-communication) is Airflow's mechanism for passing small data between tasks. The training task pushes metrics, and the evaluation task pulls them.

**Why `catchup=False`?** Without this, Airflow would try to run the DAG for every missed schedule interval since `start_date`. Since this is manual-trigger only, catchup doesn't apply, but it's a good practice to always set.

### Step 9.2: Continuous Training DAG

Create `dags/continuous_training.py`:

```python
"""
Continuous Training DAG — Drift-triggered retraining.

This DAG monitors for data drift and triggers retraining when drift is detected:
1. Check for data drift using Evidently
2. If drift detected → trigger the training_pipeline DAG
3. If no drift → skip retraining

Schedule: Weekly (every Monday at 6 AM UTC)
Can also be triggered manually.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift_task(**kwargs):
    """Check for data/model drift using Evidently.

    Compares current production data against the reference dataset
    (training data saved as reference.parquet).

    Returns 'trigger_retrain' if drift is detected, 'skip_retrain' otherwise.
    """
    import os
    import pandas as pd

    project_root = os.getenv("PROJECT_ROOT", "/opt/airflow")
    reference_path = os.path.join(
        project_root, "data", "processed", "reference.parquet"
    )
    current_path = os.path.join(project_root, "data", "processed", "train.parquet")

    # Check if required files exist
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        print("Reference or current data not found. Skipping drift check.")
        return "skip_retrain"

    try:
        from monitoring.drift_detector import check_drift

        reference_df = pd.read_parquet(reference_path)
        current_df = pd.read_parquet(current_path)

        drift_result = check_drift(reference_df, current_df)

        if drift_result["drift_detected"]:
            print(
                f"DRIFT DETECTED: {drift_result['drift_share']:.2%} of features drifted"
            )
            return "trigger_retrain"
        else:
            print(
                f"No significant drift: {drift_result['drift_share']:.2%} of features drifted"
            )
            return "skip_retrain"

    except Exception as e:
        print(f"Drift check failed: {e}. Skipping retrain.")
        return "skip_retrain"


with DAG(
    dag_id="continuous_training",
    default_args=default_args,
    description="Monitor drift and trigger retraining when needed",
    schedule="0 6 * * 1",  # Every Monday at 6 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "monitoring", "ct"],
) as dag:
    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift_task,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    check_drift >> [trigger_retrain, skip_retrain] >> done
```

**Why `BranchPythonOperator`?** It conditionally routes the workflow. Based on the return value (`"trigger_retrain"` or `"skip_retrain"`), Airflow follows only that branch and skips the other. This avoids unnecessary retraining when data hasn't changed.

**Why `trigger_rule="none_failed_min_one_success"`?** The `done` task needs to run regardless of which branch was taken. The default trigger rule (`all_success`) would fail because only one branch runs (the other is skipped). This custom rule says "run as long as nothing failed and at least one upstream succeeded."

### Step 9.3: Feature Materialization DAG

Create `dags/feature_materialization.py`:

```python
"""
Feature Materialization DAG — Incremental Feast sync.

Materializes features from the offline store (Parquet files) to the
online store (Redis) on an hourly schedule. This ensures the serving
layer always has fresh feature values for real-time predictions.

Schedule: Every hour
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def materialize_features_task(**kwargs):
    """Run incremental feature materialization.

    Materializes features from the last hour to Redis.
    Uses the execution date to determine the time window.
    """
    from features.feast_client import materialize

    # Materialize the last 2 hours to ensure overlap and no gaps
    execution_date = kwargs.get("logical_date", datetime.now())
    end_date = execution_date.isoformat()
    start_date = (execution_date - timedelta(hours=2)).isoformat()

    print(f"Materializing features from {start_date} to {end_date}")
    materialize(start_date=start_date, end_date=end_date)
    print("Materialization complete.")


with DAG(
    dag_id="feature_materialization",
    default_args=default_args,
    description="Hourly incremental feature materialization to Redis",
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features", "feast"],
) as dag:
    materialize = PythonOperator(
        task_id="materialize_features",
        python_callable=materialize_features_task,
    )
```

**Why hourly?** In a production system, new customer data arrives continuously. Hourly materialization ensures the online store (Redis) has reasonably fresh features for real-time predictions. The 2-hour window (with 1-hour overlap) prevents gaps if a run is slightly delayed.

---

## Part 10: CI/CD/CT with GitHub Actions

### Step 10.1: Test Workflow (CI)

Create `.github/workflows/test.yml`:

```yaml
name: CI — Test & Validate

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install ruff
        run: pip install ruff
      - name: Lint
        run: ruff check src/ tests/ dags/
      - name: Format check
        run: ruff format --check src/ tests/ dags/

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
          pip install pytest-cov
      - name: Run unit tests with coverage
        run: |
          PYTHONPATH=src pytest tests/unit/ -v --tb=short \
            --cov=src --cov-report=xml --cov-report=term-missing
        env:
          MLFLOW_TRACKING_URI: "http://localhost:5001"
      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage.xml

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: mlops
          POSTGRES_PASSWORD: mlops_password
          POSTGRES_DB: mlops
        ports:
          - 5432:5432
        options: --health-cmd "pg_isready -U mlops" --health-interval 10s --health-timeout 5s --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run integration tests
        run: PYTHONPATH=src pytest tests/integration/ -v --tb=short
        env:
          REDIS_HOST: localhost
          REDIS_PORT: 6379
          MLFLOW_TRACKING_URI: "http://localhost:5001"
          DATABASE_URL: "postgresql://mlops:mlops_password@localhost:5432/mlops"

  docker-build:
    name: Validate Docker Build
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build API image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.api
          push: false
          tags: mlops-api:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Build MLflow image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.mlflow
          push: false
          tags: mlops-mlflow:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install safety
        run: pip install safety pip-audit
      - name: Check for known vulnerabilities
        run: pip-audit -r requirements.txt --ignore-vuln PYSEC-2024-* || true
      - name: Check for secrets in code
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified
```

**Why `concurrency: cancel-in-progress: true`?** If you push 3 commits in quick succession, this cancels the previous CI runs and only runs the latest one. Saves CI minutes and avoids confusing results.

**Why integration tests need `services`?** Integration tests hit real databases. GitHub Actions can spin up Redis and PostgreSQL containers as part of the workflow, giving you a clean test environment without Docker Compose.

### Step 10.2: Deploy Workflow (CD)

Create `.github/workflows/deploy.yml`:

```yaml
name: CD — Build, Push & Deploy

on:
  push:
    branches: [main]
    paths-ignore:
      - "**.md"
      - "notebooks/**"
      - ".github/workflows/ct.yml"
  workflow_dispatch:
    inputs:
      environment:
        description: "Deployment environment"
        required: true
        default: "staging"
        type: choice
        options:
          - staging
          - production

concurrency:
  group: deploy-${{ github.ref }}
  cancel-in-progress: false

env:
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ghcr.io/${{ github.repository_owner }}/mlops

jobs:
  # ── Build & Push Docker Images ──────────────────────────────
  build-and-push:
    name: Build & Push Images
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    strategy:
      matrix:
        include:
          - image: api
            dockerfile: docker/Dockerfile.api
          - image: mlflow
            dockerfile: docker/Dockerfile.mlflow
          - image: airflow
            dockerfile: docker/Dockerfile.airflow
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_PREFIX }}-${{ matrix.image }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest,enable={{is_default_branch}}
            type=raw,value={{date 'YYYYMMDD-HHmmss'}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ${{ matrix.dockerfile }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Deploy to Local Machine (self-hosted runner) ───────────
  deploy-staging:
    name: Deploy to Staging
    runs-on: self-hosted
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Log in to GHCR
        run: echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin

      - name: Deploy with Docker Compose
        run: |
          cd /Users/dip/projects/mlops_project

          export IMAGE_PREFIX=${{ env.IMAGE_PREFIX }}
          export IMAGE_TAG=latest

          # Stop existing dev containers and redeploy with production images
          docker compose down --remove-orphans || true

          # Pull latest GHCR images
          docker compose -f docker-compose.yml -f docker-compose.prod.yml pull

          # Deploy with production overrides
          docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --remove-orphans

          # Wait for services to be healthy (API depends on MLflow which takes ~2 min)
          echo "Waiting for services to start..."
          for i in $(seq 1 12); do
            if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
              echo "API is healthy!"
              curl -s http://localhost:8000/health
              echo ""
              echo "Deployment successful!"
              exit 0
            fi
            echo "Waiting for API... (attempt $i/12)"
            sleep 15
          done
          echo "API did not become healthy in time"
          docker ps
          docker logs mlops_project-api-1 --tail 20
          exit 1
```

**Why `paths-ignore: "**.md"`?** Documentation-only changes shouldn't trigger a deployment. No point rebuilding Docker images when you just updated the README.

**Why `cancel-in-progress: false` for deploy?** Unlike CI (where cancelling is fine), you don't want to cancel a deployment halfway through. That could leave the system in a broken state.

**Why GHCR (GitHub Container Registry)?** Free for public repositories, integrated authentication via `GITHUB_TOKEN`, and co-located with your code on GitHub.

### Step 10.3: Continuous Training Workflow (CT)

Create `.github/workflows/ct.yml`:

```yaml
name: CT — Continuous Training

on:
  schedule:
    - cron: "0 6 * * 1"  # Every Monday at 6 AM UTC
  workflow_dispatch:
    inputs:
      force_retrain:
        description: "Force retraining even if no drift detected"
        required: false
        default: false
        type: boolean

concurrency:
  group: continuous-training
  cancel-in-progress: false

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ghcr.io/${{ github.repository_owner }}/mlops

jobs:
  # ── Step 1: Drift Detection ─────────────────────────────────
  detect-drift:
    name: Detect Data Drift
    runs-on: ubuntu-latest
    outputs:
      drift_detected: ${{ steps.drift.outputs.drift_detected }}
      drift_share: ${{ steps.drift.outputs.drift_share }}
      should_retrain: ${{ steps.decide.outputs.should_retrain }}
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Check for drift
        id: drift
        run: |
          PYTHONPATH=src python -c "
          import pandas as pd
          import os, json

          ref_path = 'data/processed/reference.parquet'
          cur_path = 'data/processed/train.parquet'

          if not os.path.exists(ref_path) or not os.path.exists(cur_path):
              print('No processed data found.')
              with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                  f.write('drift_detected=true\n')
                  f.write('drift_share=1.0\n')
          else:
              from monitoring.drift_detector import check_drift
              ref = pd.read_parquet(ref_path)
              cur = pd.read_parquet(cur_path)
              result = check_drift(ref, cur)

              print(f'Drift detected: {result[\"drift_detected\"]}')
              print(f'Drift share: {result[\"drift_share\"]:.2%}')
              print(f'Per-feature scores: {json.dumps(result[\"feature_scores\"], indent=2)}')

              with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                  f.write(f'drift_detected={str(result[\"drift_detected\"]).lower()}\n')
                  f.write(f'drift_share={result[\"drift_share\"]}\n')
          "

      - name: Decide on retraining
        id: decide
        run: |
          FORCE="${{ github.event.inputs.force_retrain }}"
          DRIFT="${{ steps.drift.outputs.drift_detected }}"
          if [ "$FORCE" = "true" ] || [ "$DRIFT" = "true" ]; then
            echo "should_retrain=true" >> "$GITHUB_OUTPUT"
            echo "Will retrain (force=$FORCE, drift=$DRIFT)"
          else
            echo "should_retrain=false" >> "$GITHUB_OUTPUT"
            echo "Skipping retrain — no drift detected"
          fi

      - name: Generate drift report
        if: always()
        run: |
          PYTHONPATH=src python -c "
          import os, pandas as pd
          ref_path = 'data/processed/reference.parquet'
          cur_path = 'data/processed/train.parquet'
          if os.path.exists(ref_path) and os.path.exists(cur_path):
              from monitoring.drift_detector import generate_drift_report
              ref = pd.read_parquet(ref_path)
              cur = pd.read_parquet(cur_path)
              generate_drift_report(ref, cur, 'drift_report.html')
          " || true

      - name: Upload drift report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: drift-report
          path: drift_report.html
          retention-days: 30

  # ── Step 2: Retrain Model ───────────────────────────────────
  retrain:
    name: Retrain Model
    runs-on: ubuntu-latest
    needs: detect-drift
    if: needs.detect-drift.outputs.should_retrain == 'true'
    outputs:
      model_version: ${{ steps.train.outputs.model_version }}
      f1_score: ${{ steps.train.outputs.f1_score }}
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: mlops
          POSTGRES_PASSWORD: mlops_password
          POSTGRES_DB: mlops
        ports:
          - 5432:5432
        options: --health-cmd "pg_isready -U mlops" --health-interval 10s --health-timeout 5s --health-retries 5
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Start MLflow server
        run: |
          mlflow server \
            --backend-store-uri postgresql://mlops:mlops_password@localhost:5432/mlops \
            --default-artifact-root ./mlartifacts \
            --host 0.0.0.0 --port 5001 \
            --allowed-hosts "*" &
          sleep 10
          curl -sf http://localhost:5001/health
        env:
          MLFLOW_TRACKING_URI: http://localhost:5001

      - name: Validate raw data
        run: |
          PYTHONPATH=src python -c "
          from data.load import load_raw_data
          from validation.validate import validate_raw_data
          df = load_raw_data()
          result = validate_raw_data(df)
          if not result['success']:
              raise ValueError(f'Validation failed: {result}')
          print('Data validation passed')
          "

      - name: Preprocess data
        run: PYTHONPATH=src python -m data.load

      - name: Train model
        id: train
        run: |
          PYTHONPATH=src python -c "
          import os, json
          from models.train import train
          model, metrics = train(tracking_uri='http://localhost:5001')
          print(f'Metrics: {json.dumps(metrics, indent=2)}')
          with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
              f.write(f'f1_score={metrics.get(\"test_f1\", metrics.get(\"f1\", 0))}\n')
              f.write(f'model_version=latest\n')
          "
        env:
          MLFLOW_TRACKING_URI: http://localhost:5001

      - name: Upload training artifacts
        uses: actions/upload-artifact@v4
        with:
          name: training-artifacts
          path: |
            data/processed/*.parquet
            mlartifacts/
          retention-days: 90

  # ── Step 3: Validate & Promote ──────────────────────────────
  validate-model:
    name: Validate & Promote Model
    runs-on: ubuntu-latest
    needs: retrain
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: mlops
          POSTGRES_PASSWORD: mlops_password
          POSTGRES_DB: mlops
        ports:
          - 5432:5432
        options: --health-cmd "pg_isready -U mlops" --health-interval 10s --health-timeout 5s --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Download training artifacts
        uses: actions/download-artifact@v4
        with:
          name: training-artifacts

      - name: Start MLflow server
        run: |
          mlflow server \
            --backend-store-uri postgresql://mlops:mlops_password@localhost:5432/mlops \
            --default-artifact-root ./mlartifacts \
            --host 0.0.0.0 --port 5001 \
            --allowed-hosts "*" &
          sleep 10

      - name: Evaluate and promote model
        run: |
          PYTHONPATH=src python -c "
          from models.evaluate import compare_champion_challenger
          import mlflow
          mlflow.set_tracking_uri('http://localhost:5001')

          # Get latest run metrics
          client = mlflow.tracking.MlflowClient()
          experiment = client.get_experiment_by_name('telco-churn')
          if experiment:
              runs = client.search_runs(experiment.experiment_id, order_by=['start_time DESC'], max_results=1)
              if runs:
                  metrics = runs[0].data.metrics
                  print(f'Latest run metrics: {metrics}')
                  result = compare_champion_challenger(metrics, tracking_uri='http://localhost:5001')
                  print(f'Promotion decision: {result[\"reason\"]}')
              else:
                  print('No runs found')
          else:
              print('No experiment found — first training run')
          "
        env:
          MLFLOW_TRACKING_URI: http://localhost:5001

  # ── Step 4: Redeploy if Model Changed ──────────────────────
  trigger-deploy:
    name: Trigger Deployment
    runs-on: ubuntu-latest
    needs: validate-model
    steps:
      - name: Trigger deploy workflow
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: 'deploy.yml',
              ref: 'main',
              inputs: { environment: 'staging' }
            });
            console.log('Deployment triggered after successful retraining');

  # ── Notify ──────────────────────────────────────────────────
  notify:
    name: Notify Results
    runs-on: ubuntu-latest
    needs: [detect-drift, retrain, validate-model]
    if: always()
    steps:
      - name: Summary
        run: |
          echo "## CT Pipeline Results" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "| Stage | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|-------|--------|" >> $GITHUB_STEP_SUMMARY
          echo "| Drift Detection | ${{ needs.detect-drift.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Drift Detected | ${{ needs.detect-drift.outputs.drift_detected }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Drift Share | ${{ needs.detect-drift.outputs.drift_share }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Retraining | ${{ needs.retrain.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| F1 Score | ${{ needs.retrain.outputs.f1_score }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Validation | ${{ needs.validate-model.result }} |" >> $GITHUB_STEP_SUMMARY
```

**Why both an Airflow DAG AND a GitHub Actions CT workflow?** They serve different purposes:
- **Airflow CT DAG** runs *inside* the Docker stack. It checks drift and triggers retraining within the existing infrastructure. Good for internal monitoring.
- **GitHub Actions CT workflow** runs *externally* in GitHub's cloud. It can rebuild Docker images, push to GHCR, and redeploy the entire stack. Good for the full CI/CD/CT pipeline.

---

## Part 11: Testing & Makefile

### Step 11.1: Makefile

Create `Makefile`:

```makefile
.PHONY: setup docker-up docker-down train serve test lint feast-apply feast-materialize clean

# ——— Setup ———
setup:
	cp -n .env.example .env || true
	pip install -r requirements.txt
	pip install -e .

# ——— Docker ———
docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# ——— Data & Training ———
train:
	python -m models.train

preprocess:
	python -m data.load

validate:
	python -m validation.validate

# ——— Feature Store ———
feast-apply:
	docker compose exec api python -c "\
		import importlib.util, os; \
		os.chdir('/app/config/feast'); \
		spec = importlib.util.spec_from_file_location('features', '/app/config/feast/features.py'); \
		mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
		from feast import FeatureStore; store = FeatureStore(repo_path='/app/config/feast'); \
		store.apply([mod.customer, mod.customer_demographics, mod.customer_account, mod.customer_services]); \
		print('Feast feature views applied')"

feast-materialize:
	docker compose exec api python -c "\
		from features.feast_client import get_store; \
		from datetime import datetime, timedelta; \
		store = get_store(); \
		store.materialize(start_date=datetime.now() - timedelta(days=365), end_date=datetime.now() + timedelta(days=2)); \
		print('Materialization complete')"

# ——— Serving ———
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# ——— Testing ———
test:
	PYTHONPATH=src pytest tests/ -v

test-unit:
	PYTHONPATH=src pytest tests/unit/ -v

test-integration:
	PYTHONPATH=src pytest tests/integration/ -v

# ——— Linting ———
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

# ——— Cleanup ———
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
```

**Why a Makefile?** It provides a consistent, memorable interface. `make train` is easier to remember than `PYTHONPATH=src python -m models.train`. It also documents the available commands — run `make` with no arguments to see them all.

### Step 11.2: Tests

Create test files in `tests/unit/` and `tests/integration/` matching the source modules. Unit tests mock external services (MLflow, Redis) and run fast. Integration tests require Docker services and verify end-to-end behavior.

Create `tests/conftest.py` with shared fixtures (sample data, mocked clients).

---

## Part 12: Running the Complete System

### Step 12.1: First-Time Startup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Start all services
make docker-up

# This builds 3 Docker images and starts 10 containers.
# First run takes 5-10 minutes for image builds.
# Subsequent runs are fast (cached layers).

# 3. Check service status
make docker-ps

# You should see all services with status "Up" or "healthy"
# airflow-init will show "Exited (0)" — that's expected (it runs once and exits)
```

Wait 1-2 minutes for all health checks to pass. Then:

```bash
# 4. Check service UIs
#    MLflow:   http://localhost:5001
#    Airflow:  http://localhost:8080  (admin/admin)
#    Grafana:  http://localhost:3000  (admin/admin)
#    API docs: http://localhost:8000/docs

# 5. Run the training pipeline (first time — creates the model)
make train

# 6. Apply Feast feature definitions and materialize to Redis
make feast-apply
make feast-materialize

# 7. Test the API
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "TEST-001",
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
    "TotalCharges": 29.85
  }'
```

Expected response:
```json
{
  "customerID": "TEST-001",
  "churn_prediction": 1,
  "churn_probability": 0.7234,
  "model_version": "1"
}
```

### Step 12.2: Verification Checklist

| Service | URL | What to Check |
|---------|-----|---------------|
| **API** | http://localhost:8000/health | `{"status": "healthy", "model_loaded": true}` |
| **API Docs** | http://localhost:8000/docs | Swagger UI with all endpoints |
| **MLflow** | http://localhost:5001 | "telco-churn-experiment" with at least 1 run |
| **Airflow** | http://localhost:8080 | 3 DAGs visible: training_pipeline, continuous_training, feature_materialization |
| **Grafana** | http://localhost:3000 | MLOps Overview dashboard with panels |
| **Prometheus** | http://localhost:9090 | Targets page shows `fastapi` as `UP` |
| **Redis** | http://localhost:5540 | RedisInsight shows connected databases |

### Step 12.3: Troubleshooting

**Port 5000 conflict on macOS:**
macOS uses port 5000 for AirPlay Receiver. The `docker-compose.yml` already maps MLflow to port 5001 externally. If you still have issues, disable AirPlay Receiver in System Settings → General → AirDrop & Handoff.

**Docker out of memory:**
The full stack needs ~4 GB RAM. Open Docker Desktop → Settings → Resources → Memory and increase to at least 4 GB. If still struggling, stop `redisinsight` (optional service).

**Airflow init fails:**
Check logs with `docker compose logs airflow-init`. Common issue: the `airflow` database doesn't exist in PostgreSQL. Verify `docker/init-db.sh` is executable (`chmod +x docker/init-db.sh`) and mounted correctly.

**Model not loading (API returns 503):**
This is expected before the first training run. Run `make train` to create the initial model. After training, the API will load it within ~30 seconds.

**Feast materialization fails:**
Ensure the processed data exists (`data/processed/train.parquet`). If not, run `make preprocess` or `make train` first.

---

## Part 13: Architecture Recap

### System Architecture

```
                    GitHub Actions (CI/CD/CT)
                           │
    ┌──────────────────────┼──────────────────────────┐
    │                      ▼                          │
    │  ┌─────────┐   ┌──────────┐   ┌─────────────┐  │
    │  │  DVC    │──▶│ Airflow  │──▶│   MLflow     │  │
    │  │ (data)  │   │ (DAGs)   │   │ (tracking +  │  │
    │  └─────────┘   └────┬─────┘   │  registry)   │  │
    │                     │         └──────┬──────┘  │
    │  ┌─────────┐        │                │         │
    │  │  Great  │◀───────┘         ┌──────▼──────┐  │
    │  │  Expect │                  │  FastAPI    │  │
    │  └─────────┘   ┌─────────┐   │  (serving)  │  │
    │                │  Feast   │──▶│  :8000      │  │
    │                │ (features│   └──────┬──────┘  │
    │                │  Redis)  │          │         │
    │                └─────────┘   ┌──────▼──────┐  │
    │                              │ Prometheus  │  │
    │  ┌─────────┐                 │  + Grafana  │  │
    │  │Evidently│────────────────▶│ (monitoring)│  │
    │  │ (drift) │                 └─────────────┘  │
    │  └─────────┘                                   │
    └────────────────────────────────────────────────┘
```

### Data Flow: Prediction Request

1. Client sends POST to `/predict` with raw customer data
2. FastAPI validates the request using Pydantic schemas
3. `compute_features()` engineers features (tenure bucket, service count, etc.)
4. `ChurnPredictor._align_features()` matches columns to what the model expects
5. XGBoost model (loaded from MLflow) makes prediction
6. Prometheus middleware records the request latency and prediction
7. Response returned with prediction, probability, and model version

### Data Flow: Training Pipeline

1. Airflow `continuous_training` DAG runs weekly
2. Evidently compares current data against reference data
3. If drift detected → triggers `training_pipeline` DAG
4. Pipeline: validate → preprocess → feast apply → materialize → train → evaluate
5. MLflow logs all metrics, parameters, and the model artifact
6. If new model beats champion by >1% F1 → promote to champion
7. API picks up the new champion on next model load

### Data Flow: Deployment

1. Developer pushes code to `main` branch
2. GitHub Actions `test.yml` runs: lint → unit tests → integration tests → docker build → security scan
3. If all pass, `deploy.yml` triggers: build images → push to GHCR → deploy via Docker Compose
4. Health check loop verifies the API is responding
5. Weekly, `ct.yml` checks for drift and conditionally retrains + redeploys
