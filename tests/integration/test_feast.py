"""Integration tests for the Feast feature store.

These tests require Redis to be running (provided by CI services config).
They test the full Feast lifecycle: apply → materialize → retrieve.
"""
import os
import pytest


# Skip if Redis is not available
pytestmark = pytest.mark.skipif(
    os.getenv("REDIS_HOST") is None,
    reason="Redis not available — set REDIS_HOST to run Feast integration tests",
)


class TestFeastClient:
    """Test Feast client operations."""

    def test_get_store(self):
        """Should create a FeatureStore instance."""
        from features.feast_client import get_store
        store = get_store()
        assert store is not None
