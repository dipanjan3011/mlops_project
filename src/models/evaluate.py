"""
Model evaluation and champion/challenger comparison.

Computes comprehensive metrics and decides whether a newly trained model
(challenger) should replace the current champion.
"""
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.hyperparams import MODEL_NAME, CHAMPION_ALIAS, CHALLENGER_ALIAS

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
