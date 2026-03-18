"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError


class TestCustomerInput:
    """Test the CustomerInput schema."""

    def test_valid_input(self, sample_customer_input):
        """Valid customer data should pass validation."""
        from serving.schemas import CustomerInput

        customer = CustomerInput(**sample_customer_input)
        assert customer.customerID == "TEST-001"
        assert customer.tenure == 12

    def test_missing_required_field(self):
        """Missing required fields should raise ValidationError."""
        from serving.schemas import CustomerInput

        with pytest.raises(ValidationError):
            CustomerInput(gender="Male")  # Missing many required fields

    def test_invalid_senior_citizen(self, sample_customer_input):
        """SeniorCitizen must be 0 or 1."""
        from serving.schemas import CustomerInput

        sample_customer_input["SeniorCitizen"] = 5
        with pytest.raises(ValidationError):
            CustomerInput(**sample_customer_input)

    def test_negative_tenure(self, sample_customer_input):
        """Tenure cannot be negative."""
        from serving.schemas import CustomerInput

        sample_customer_input["tenure"] = -1
        with pytest.raises(ValidationError):
            CustomerInput(**sample_customer_input)

    def test_negative_charges(self, sample_customer_input):
        """MonthlyCharges cannot be negative."""
        from serving.schemas import CustomerInput

        sample_customer_input["MonthlyCharges"] = -10.0
        with pytest.raises(ValidationError):
            CustomerInput(**sample_customer_input)


class TestPredictionOutput:
    """Test the PredictionOutput schema."""

    def test_valid_output(self):
        """Valid prediction output should be created."""
        from serving.schemas import PredictionOutput

        output = PredictionOutput(
            customerID="TEST-001",
            churn_prediction=1,
            churn_probability=0.85,
            model_version="1",
        )
        assert output.churn_prediction == 1
        assert output.churn_probability == 0.85


class TestHealthResponse:
    """Test the HealthResponse schema."""

    def test_healthy_response(self):
        """Healthy response with model loaded."""
        from serving.schemas import HealthResponse

        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version="3",
        )
        assert health.status == "healthy"
        assert health.model_loaded is True

    def test_degraded_response(self):
        """Degraded response without model."""
        from serving.schemas import HealthResponse

        health = HealthResponse(
            status="degraded",
            model_loaded=False,
        )
        assert health.model_version is None
