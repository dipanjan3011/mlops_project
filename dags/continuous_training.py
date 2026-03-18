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
from airflow.operators.python import PythonOperator, BranchPythonOperator
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
    reference_path = os.path.join(project_root, "data", "processed", "reference.parquet")
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
            print(f"DRIFT DETECTED: {drift_result['drift_share']:.2%} of features drifted")
            return "trigger_retrain"
        else:
            print(f"No significant drift: {drift_result['drift_share']:.2%} of features drifted")
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
