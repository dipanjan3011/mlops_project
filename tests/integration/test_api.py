"""Integration tests for the FastAPI prediction API."""
import pytest


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, api_client):
        """Health endpoint should always return 200."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, api_client):
        """Health response should have required fields."""
        response = api_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    """Test the /predict endpoint."""

    def test_predict_without_model_returns_503(self, api_client, sample_customer_input):
        """Predict should return 503 when no model is loaded."""
        response = api_client.post("/predict", json=sample_customer_input)
        # Without a running MLflow server, model won't be loaded
        assert response.status_code in [503, 500]

    def test_predict_invalid_input_returns_422(self, api_client):
        """Invalid input should return 422 validation error."""
        response = api_client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Test the /batch-predict endpoint."""

    def test_batch_predict_invalid_returns_422(self, api_client):
        """Invalid batch input should return 422."""
        response = api_client.post("/batch-predict", json={"invalid": "data"})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test the /metrics endpoint."""

    def test_metrics_returns_200(self, api_client):
        """Metrics endpoint should return Prometheus format data."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        # Prometheus metrics should contain HELP/TYPE lines
        assert "api_request_total" in response.text or "python_info" in response.text
