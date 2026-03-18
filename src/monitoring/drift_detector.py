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
