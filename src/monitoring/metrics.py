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
