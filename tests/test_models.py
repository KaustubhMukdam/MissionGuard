# tests/test_models.py
"""Tests for anomaly detection models."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.models import (
    StatisticalBaseline,
    RollingMADBaseline,
    RollingZScoreBaseline,
    IsolationForestDetector,
    BaseAnomalyDetector,
)
from missionguard.models.base import BaseAnomalyDetector as BaseDetector


class TestStatisticalBaseline:
    """Tests for StatisticalBaseline detector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n) * 2,
            "feat3": np.random.randn(n) * 0.5,
        })
        return df
    
    def test_fit_predict_mad(self, sample_data):
        """Test MAD baseline fit and predict."""
        detector = StatisticalBaseline(method="mad")
        detector.fit(sample_data)
        
        assert detector.fitted
        assert detector.feature_names == ["feat1", "feat2", "feat3"]
        
        scores = detector.score(sample_data)
        assert len(scores) == len(sample_data)
        assert np.all(scores >= 0)
    
    def test_fit_predict_zscore(self, sample_data):
        """Test Z-score baseline fit and predict."""
        detector = StatisticalBaseline(method="zscore")
        detector.fit(sample_data)
        
        assert detector.fitted
        scores = detector.score(sample_data)
        assert len(scores) == len(sample_data)
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        detector = StatisticalBaseline(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            detector.fit(pd.DataFrame({"a": [1, 2, 3]}))
    
    def test_threshold_tuning(self, sample_data):
        """Test threshold tuning on validation data."""
        detector = StatisticalBaseline(method="mad")
        detector.fit(sample_data)
        
        # Create validation labels with some anomalies
        val_data = sample_data.copy()
        val_labels = pd.Series([0] * 900 + [1] * 100)
        
        # Inject anomalies in validation data
        val_data.loc[900:, "feat1"] += 5
        
        threshold = detector.tune_threshold(val_data, val_labels, metric="f1")
        assert threshold is not None
        assert threshold > 0
    
    def test_threshold_percentile(self, sample_data):
        """Test setting threshold from percentile."""
        detector = StatisticalBaseline(method="mad")
        detector.fit(sample_data)
        
        scores = detector.score(sample_data)
        detector.set_threshold_from_scores(scores, method="percentile", value=95.0)
        
        assert detector.threshold is not None
        # Should be around 95th percentile
        expected = np.percentile(scores, 95)
        assert abs(detector.threshold - expected) < 1e-6
    
    def test_predict(self, sample_data):
        """Test binary prediction."""
        detector = StatisticalBaseline(method="mad")
        detector.fit(sample_data)
        
        scores = detector.score(sample_data)
        detector.set_threshold_from_scores(scores, method="percentile", value=90.0)
        
        preds = detector.predict(sample_data)
        assert len(preds) == len(sample_data)
        assert set(preds).issubset({0, 1})
    
    def test_persistence(self, sample_data, tmp_path):
        """Test save/load."""
        detector = StatisticalBaseline(method="mad")
        detector.fit(sample_data)
        
        scores_before = detector.score(sample_data)
        detector.set_threshold_from_scores(scores_before, method="percentile", value=95.0)
        
        save_path = tmp_path / "detector.joblib"
        detector.save(save_path)
        
        loaded = StatisticalBaseline.load(save_path)
        
        scores_after = loaded.score(sample_data)
        np.testing.assert_array_almost_equal(scores_before, scores_after)
        assert loaded.threshold == detector.threshold


class TestRollingMADBaseline:
    """Tests for RollingMADBaseline detector."""
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data with trend."""
        np.random.seed(42)
        n = 500
        # Add trend
        trend = np.linspace(0, 10, n)
        noise = np.random.randn(n) * 0.5
        values = trend + noise
        
        df = pd.DataFrame({
            "feat1": values,
            "feat2": values * 0.5 + np.random.randn(n) * 0.2,
        }, index=pd.date_range("2022-01-01", periods=n, freq="1min"))
        return df
    
    def test_fit_score(self, time_series_data):
        """Test rolling MAD fit and score."""
        detector = RollingMADBaseline(window=50, min_periods=10)
        detector.fit(time_series_data)
        
        assert detector.fitted
        scores = detector.score(time_series_data)
        assert len(scores) == len(time_series_data)
    
    def test_window_parameter(self, time_series_data):
        """Test different window sizes."""
        for window in [20, 50, 100]:
            detector = RollingMADBaseline(window=window, min_periods=5)
            detector.fit(time_series_data)
            scores = detector.score(time_series_data)
            assert len(scores) == len(time_series_data)


class TestRollingZScoreBaseline:
    """Tests for RollingZScoreBaseline detector."""
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data."""
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n) * 2,
        }, index=pd.date_range("2022-01-01", periods=n, freq="1min"))
        return df
    
    def test_fit_score(self, time_series_data):
        """Test rolling Z-score fit and score."""
        detector = RollingZScoreBaseline(window=30, min_periods=5)
        detector.fit(time_series_data)
        
        assert detector.fitted
        scores = detector.score(time_series_data)
        assert len(scores) == len(time_series_data)
    
    def test_aggregation_methods(self, time_series_data):
        """Test different aggregation methods."""
        for agg in ["max", "mean", "sum"]:
            detector = RollingZScoreBaseline(window=30, aggregation=agg)
            detector.fit(time_series_data)
            scores = detector.score(time_series_data)
            assert len(scores) == len(time_series_data)


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample multivariate data."""
        np.random.seed(42)
        n = 1000
        # Normal data
        normal = np.random.randn(n, 5)
        # Add some anomalies
        anomalies = np.random.randn(50, 5) * 5
        data = np.vstack([normal, anomalies])
        
        df = pd.DataFrame(data, columns=[f"feat{i}" for i in range(5)])
        return df
    
    def test_fit_score(self, sample_data):
        """Test Isolation Forest fit and score."""
        detector = IsolationForestDetector(
            n_estimators=50,
            score_normalization="minmax",
            random_state=42,
        )
        detector.fit(sample_data)
        
        assert detector.fitted
        scores = detector.score(sample_data)
        assert len(scores) == len(sample_data)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
    
    def test_raw_scores(self, sample_data):
        """Test getting raw scores."""
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        detector.fit(sample_data)
        
        raw = detector.get_raw_scores(sample_data)
        normalized = detector.score(sample_data)
        
        # Raw scores should be different from normalized
        assert not np.allclose(raw, normalized)
    
    def test_contamination_parameter(self, sample_data):
        """Test contamination parameter."""
        for contam in [0.01, 0.05, 0.1, "auto"]:
            detector = IsolationForestDetector(
                n_estimators=30,
                contamination=contam,
                random_state=42,
            )
            detector.fit(sample_data)
            assert detector.fitted
    
    def test_score_normalization_options(self, sample_data):
        """Test different score normalization methods."""
        for norm in ["minmax", "percentile", "none"]:
            detector = IsolationForestDetector(
                n_estimators=30,
                score_normalization=norm,
                random_state=42,
            )
            detector.fit(sample_data)
            scores = detector.score(sample_data)
            
            if norm != "none":
                assert np.all(scores >= 0)
                assert np.all(scores <= 1)
    
    def test_persistence(self, sample_data, tmp_path):
        """Test save/load."""
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        detector.fit(sample_data)
        
        scores_before = detector.score(sample_data)
        detector.set_threshold_from_scores(scores_before, method="percentile", value=95.0)
        
        save_path = tmp_path / "if_detector.joblib"
        detector.save(save_path)
        
        loaded = IsolationForestDetector.load(save_path)
        
        scores_after = loaded.score(sample_data)
        np.testing.assert_array_almost_equal(scores_before, scores_after)
        assert loaded.threshold == detector.threshold
    
    def test_feature_names_tracking(self, sample_data):
        """Test feature names are tracked."""
        detector = IsolationForestDetector(n_estimators=30, random_state=42)
        detector.fit(sample_data)
        
        assert detector.feature_names == list(sample_data.columns)
        
        # Test with custom feature names
        detector2 = IsolationForestDetector(
            n_estimators=30, 
            feature_names=["feat1", "feat2"],
            random_state=42,
        )
        detector2.fit(sample_data[["feat1", "feat2"]])
        assert detector2.feature_names == ["feat1", "feat2"]


class TestBaseDetector:
    """Tests for base detector functionality."""
    
    def test_abstract_methods(self):
        """Test that BaseAnomalyDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDetector("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])