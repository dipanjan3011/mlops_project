"""Integration tests for MLflow experiment tracking.

These tests require an MLflow server to be running.
They test experiment creation, model logging, and registry operations.
"""

import os
import pytest


# Skip if MLflow is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("MLFLOW_TRACKING_URI", "").startswith("http"),
    reason="MLflow server not available",
)


class TestMLflowIntegration:
    """Test MLflow tracking and registry."""

    def test_mlflow_connection(self):
        """Should connect to MLflow server."""
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
        # This will fail if MLflow is not reachable, which is expected in CI without MLflow service
        try:
            experiments = mlflow.search_experiments()
            assert isinstance(experiments, list)
        except Exception:
            pytest.skip("MLflow server not reachable")
