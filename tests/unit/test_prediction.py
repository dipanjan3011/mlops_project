"""Tests for prediction utilities."""
import pandas as pd


class TestChurnPredictor:
    """Test the ChurnPredictor class."""

    def test_predictor_init(self):
        """Predictor should initialize without a loaded model."""
        from models.predict import ChurnPredictor
        predictor = ChurnPredictor(tracking_uri="http://fake:5000")
        assert predictor.model is None
        assert predictor.model_version is None

    def test_predict_without_model_raises(self):
        """Predict should raise if model not loaded."""
        from models.predict import ChurnPredictor
        predictor = ChurnPredictor(tracking_uri="http://fake:5000")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict(pd.DataFrame({"a": [1]}))

    def test_predict_batch_without_model_raises(self):
        """Batch predict should raise if model not loaded."""
        from models.predict import ChurnPredictor
        predictor = ChurnPredictor(tracking_uri="http://fake:5000")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict_batch(pd.DataFrame({"a": [1]}))
