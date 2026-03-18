"""Tests for feature engineering functions."""


class TestTenureBuckets:
    """Test tenure bucketing."""

    def test_tenure_bucket_created(self, sample_clean_data):
        """Should create tenure_bucket column."""
        from data.preprocess import create_tenure_buckets
        result = create_tenure_buckets(sample_clean_data)
        assert "tenure_bucket" in result.columns

    def test_bucket_values(self, sample_clean_data):
        """Bucket values should be in expected set."""
        from data.preprocess import create_tenure_buckets
        result = create_tenure_buckets(sample_clean_data)
        valid_buckets = {"0-12", "13-24", "25-48", "49-60", "61-72"}
        actual = set(result["tenure_bucket"].dropna().astype(str))
        assert actual.issubset(valid_buckets)


class TestServiceCount:
    """Test service counting."""

    def test_service_count_created(self, sample_clean_data):
        """Should create service_count column."""
        from data.preprocess import count_services
        result = count_services(sample_clean_data)
        assert "service_count" in result.columns

    def test_service_count_range(self, sample_clean_data):
        """Service count should be 0-7."""
        from data.preprocess import count_services
        result = count_services(sample_clean_data)
        assert result["service_count"].min() >= 0
        assert result["service_count"].max() <= 7


class TestChargesFeatures:
    """Test charge-related features."""

    def test_avg_monthly_charge(self, sample_clean_data):
        """Should compute avg_monthly_charge."""
        from data.preprocess import compute_charges_features
        result = compute_charges_features(sample_clean_data)
        assert "avg_monthly_charge" in result.columns
        assert result["avg_monthly_charge"].isna().sum() == 0


class TestFullPreprocessing:
    """Test the complete preprocessing pipeline."""

    def test_preprocess_runs(self, sample_clean_data):
        """Full preprocessing should run without errors."""
        from data.preprocess import preprocess_for_training
        result = preprocess_for_training(sample_clean_data)
        assert len(result) == len(sample_clean_data)

    def test_no_original_categoricals(self, sample_processed_data):
        """Original categorical columns should be one-hot encoded."""
        # After encoding, "gender" column should be gone, replaced by dummies
        assert "gender" not in sample_processed_data.columns
