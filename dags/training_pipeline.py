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
