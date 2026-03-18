"""
Model prediction utilities.

Loads the champion model from MLflow Model Registry and provides
prediction functions for single records and batches.
"""
import os
import time

import mlflow
import pandas as pd
import numpy as np

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
                version_info = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
                self.model_version = version_info.version

                # Load the model
                model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
                self.model = mlflow.xgboost.load_model(model_uri)

                # Load feature columns
                run = client.get_run(version_info.run_id)
                artifact_uri = run.info.artifact_uri
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
                    wait = retry_delay * (2 ** attempt)
                    print(f"Model load attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
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
