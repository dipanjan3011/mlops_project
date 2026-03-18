"""Tests for data loading and cleaning."""
import pandas as pd
import pytest


class TestLoadRawData:
    """Test raw data loading."""

    def test_load_returns_dataframe(self, sample_raw_data):
        """Raw data should be a DataFrame."""
        assert isinstance(sample_raw_data, pd.DataFrame)

    def test_expected_columns(self, sample_raw_data):
        """Raw data should have 21 columns."""
        assert len(sample_raw_data.columns) == 21

    def test_has_customer_id(self, sample_raw_data):
        """Raw data should have customerID column."""
        assert "customerID" in sample_raw_data.columns


class TestCleanData:
    """Test data cleaning functions."""

    def test_total_charges_numeric(self, sample_clean_data):
        """TotalCharges should be numeric after cleaning."""
        assert sample_clean_data["TotalCharges"].dtype in ["float64", "float32"]

    def test_total_charges_whitespace_handled(self, sample_raw_data):
        """Whitespace in TotalCharges should be converted to 0.0."""
        from data.load import clean_data
        cleaned = clean_data(sample_raw_data)
        # Row with tenure=0 had TotalCharges=" "
        tenure_zero = cleaned[cleaned["tenure"] == 0]
        assert (tenure_zero["TotalCharges"] == 0.0).all()

    def test_no_null_total_charges(self, sample_clean_data):
        """TotalCharges should have no nulls after cleaning."""
        assert sample_clean_data["TotalCharges"].isna().sum() == 0

    def test_churn_encoded(self, sample_clean_data):
        """Churn should be encoded as 0/1 integers."""
        assert set(sample_clean_data["Churn"].unique()).issubset({0, 1})
