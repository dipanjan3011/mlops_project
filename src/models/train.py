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
