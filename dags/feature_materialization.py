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
