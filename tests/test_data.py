# tests/test_data.py
"""Tests for data loading and validation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.data.loaders import (
    load_segments,
    load_dataset,
    get_train_test_split,
    load_opssat_ad,
    get_temporal_train_test_split,
)
from missionguard.data.schema import (
    SegmentsSchema,
    DatasetSchema,
    validate_segments_df,
    validate_dataset_df,
)


class TestSegmentsSchema:
    """Tests for segments schema validation."""
    
    def test_valid_segments_df(self):
        """Test that valid OPSSAT-AD segments pass validation."""
        df = pd.DataFrame({
            "channel": ["CADC0872", "CADC0872"],
            "timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "value": [1.0, 2.0],
            "label": ["nominal", "anomaly"],
            "sampling": [1, 1],
            "anomaly": [0, 1],
            "segment": [1, 2],
            "train": [1, 0],
        })
        
        result = validate_segments_df(df)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_missing_columns(self):
        """Test that missing columns are caught."""
        df = pd.DataFrame({
            "channel": ["CADC0872"],
            "value": [1.0],
        })
        
        result = validate_segments_df(df)
        assert result["valid"] is False
        assert any("Missing required columns" in e for e in result["errors"])
    
    def test_invalid_anomaly_values(self):
        """Test that non-binary anomaly values are caught."""
        df = pd.DataFrame({
            "channel": ["CADC0872"],
            "timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "value": [1.0],
            "label": ["nominal"],
            "sampling": [1],
            "anomaly": [2],  # Invalid!
            "segment": [1],
            "train": [1],
        })
        
        result = validate_segments_df(df)
        assert result["valid"] is False
        assert any("Invalid anomaly values" in e for e in result["errors"])
    
    def test_invalid_train_values(self):
        """Test that non-binary train values are caught."""
        df = pd.DataFrame({
            "channel": ["CADC0872"],
            "timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "value": [1.0],
            "label": ["nominal"],
            "sampling": [1],
            "anomaly": [0],
            "segment": [1],
            "train": [2],  # Invalid!
        })
        
        result = validate_segments_df(df)
        assert result["valid"] is False
        assert any("Invalid train values" in e for e in result["errors"])
    
    def test_negative_segment_ids(self):
        """Test that negative segment IDs are caught."""
        df = pd.DataFrame({
            "channel": ["CADC0872"],
            "timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "value": [1.0],
            "label": ["nominal"],
            "sampling": [1],
            "anomaly": [0],
            "segment": [-1],  # Invalid!
            "train": [1],
        })
        
        result = validate_segments_df(df)
        assert result["valid"] is False
        assert any("Segment IDs must be positive" in e for e in result["errors"])
    
    def test_missing_values(self):
        """Test that missing values are caught."""
        df = pd.DataFrame({
            "channel": ["CADC0872", None],
            "timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "value": [1.0, 2.0],
            "label": ["nominal", "anomaly"],
            "sampling": [1, 1],
            "anomaly": [0, 1],
            "segment": [1, 2],
            "train": [1, 0],
        })
        
        result = validate_segments_df(df)
        assert result["valid"] is False
        assert any("Missing values found" in e for e in result["errors"])


class TestDatasetSchema:
    """Tests for dataset schema validation."""
    
    def test_valid_dataset_df(self):
        """Test that valid OPSSAT-AD dataset passes validation."""
        feature_cols = [
            "duration", "len", "mean", "var", "std", "kurtosis", "skew",
            "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
            "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
            "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"
        ]
        
        data = {
            "segment": [1, 2],
            "anomaly": [1, 0],
            "train": [1, 0],
            "channel": ["CADC0872", "CADC0872"],
            "sampling": [1, 1],
        }
        for col in feature_cols:
            data[col] = [1.0, 2.0]
        
        df = pd.DataFrame(data)
        result = validate_dataset_df(df)
        assert result["valid"] is True
    
    def test_duplicate_segments(self):
        """Test that duplicate segment IDs are caught."""
        feature_cols = ["duration", "len", "mean"]
        
        data = {
            "segment": [1, 1],  # Duplicate!
            "anomaly": [1, 0],
            "train": [1, 0],
            "channel": ["CADC0872", "CADC0872"],
            "sampling": [1, 1],
        }
        for col in feature_cols:
            data[col] = [1.0, 2.0]
        
        df = pd.DataFrame(data)
        result = validate_dataset_df(df)
        assert result["valid"] is False
        assert any("Duplicate segment IDs" in e for e in result["errors"])
    
    def test_inf_in_features(self):
        """Test that infinite values in features generate warning."""
        feature_cols = ["duration", "len", "mean"]
        
        data = {
            "segment": [1, 2],
            "anomaly": [1, 0],
            "train": [1, 0],
            "channel": ["CADC0872", "CADC0872"],
            "sampling": [1, 1],
            "duration": [1.0, np.inf],  # Infinite!
            "len": [10.0, 20.0],
            "mean": [1.0, 2.0],
            "var": [1.0, 2.0],
            "std": [1.0, 1.414],
            "kurtosis": [0.0, 0.0],
            "skew": [0.0, 0.0],
            "n_peaks": [1, 1],
            "smooth10_n_peaks": [1, 1],
            "smooth20_n_peaks": [1, 1],
            "diff_peaks": [1, 1],
            "diff2_peaks": [1, 1],
            "diff_var": [1.0, 1.0],
            "diff2_var": [1.0, 1.0],
            "gaps_squared": [1.0, 1.0],
            "len_weighted": [10.0, 20.0],
            "var_div_duration": [1.0, 0.1],
            "var_div_len": [0.1, 0.1],
        }
        
        df = pd.DataFrame(data)
        result = validate_dataset_df(df)
        assert result["valid"] is True  # Warnings don't invalidate
        assert any("infinite values" in w for w in result["warnings"])


class TestLoaders:
    """Tests for data loaders."""
    
    @pytest.fixture
    def data_dir(self):
        """Path to OPSSAT-AD data."""
        return Path(__file__).parent.parent / "data" / "raw" / "opssat-ad"
    
    def test_load_segments(self, data_dir):
        """Test loading segments.csv."""
        df = load_segments(data_dir / "segments.csv", validate=True)
        
        assert len(df) == 303493
        assert "timestamp" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["anomaly"].isin([0, 1]).all()
        assert df["train"].isin([0, 1]).all()
    
    def test_load_dataset(self, data_dir):
        """Test loading dataset.csv."""
        df = load_dataset(data_dir / "dataset.csv", validate=True)
        
        assert len(df) == 2123
        assert "segment" in df.columns
        assert df["anomaly"].isin([0, 1]).all()
        assert df["train"].isin([0, 1]).all()
    
    def test_get_train_test_split(self, data_dir):
        """Test train/test split from segments."""
        segments = load_segments(data_dir / "segments.csv", validate=False)
        dataset = load_dataset(data_dir / "dataset.csv", validate=False)
        
        train_seg, test_seg, train_ds, test_ds = get_train_test_split(segments, dataset)
        
        assert len(train_seg) == 225178
        assert len(test_seg) == 78315
        assert len(train_ds) == 1594
        assert len(test_ds) == 529
        
        # Check no overlap
        assert set(train_seg["segment"].unique()).isdisjoint(set(test_seg["segment"].unique()))
    
    def test_load_opssat_ad(self, data_dir):
        """Test convenience loader for OPSSAT-AD."""
        segments, dataset = load_opssat_ad(data_dir, validate=True)
        
        assert len(segments) == 303493
        assert len(dataset) == 2123


class TestTemporalSplit:
    """Tests for temporal train/test splitting."""
    
    def test_temporal_split_basic(self):
        """Test basic temporal split functionality."""
        dates = pd.date_range("2022-01-01", periods=100, freq="1H", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.randn(100),
            "anomaly": [0] * 100,
        })
        
        train, test = get_temporal_train_test_split(df, test_ratio=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
        
        # Train should be earlier than test
        assert train["timestamp"].max() <= test["timestamp"].min()
    
    def test_temporal_split_preserves_order(self):
        """Test that temporal split preserves chronological order."""
        dates = pd.date_range("2022-01-01", periods=50, freq="1H", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.arange(50),  # Increasing values
        })
        
        train, test = get_temporal_train_test_split(df, test_ratio=0.4)
        
        # All train values should be less than test values
        assert train["value"].max() < test["value"].min()


class TestPreprocessingTransforms:
    """Tests for preprocessing transforms."""
    
    def test_standard_scaler_wrapper(self):
        """Test StandardScalerWrapper fit/transform."""
        from missionguard.preprocessing.transforms import StandardScalerWrapper
        
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 0, 1, 1, 1],
        })
        
        scaler = StandardScalerWrapper(["feature1", "feature2"])
        scaler.fit(df)
        
        transformed = scaler.transform(df)
        
        # Check mean ~0
        assert abs(transformed["feature1"].mean()) < 1e-10
        assert abs(transformed["feature2"].mean()) < 1e-10
        
        # Check std ~1 (using sample std, ddof=1)
        # For [1,2,3,4,5], sample std = sqrt(10/4) = 1.581, scaled std = 1.0
        # But StandardScaler uses population std by default, so scaled std = 1/sqrt((n-1)/n) = sqrt(5/4) = 1.118
        # Actually sklearn's StandardScaler uses population std (ddof=0)
        # Pandas std uses ddof=1 by default
        assert abs(transformed["feature1"].std(ddof=0) - 1.0) < 1e-10
        assert abs(transformed["feature2"].std(ddof=0) - 1.0) < 1e-10
    
    def test_robust_scaler_wrapper(self):
        """Test RobustScalerWrapper fit/transform."""
        from missionguard.preprocessing.transforms import RobustScalerWrapper
        
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 0, 1, 1, 1],
        })
        
        scaler = RobustScalerWrapper(["feature1", "feature2"])
        scaler.fit(df)
        
        transformed = scaler.transform(df)
        
        # Robust scaler uses median and IQR
        # Just check it runs without error and preserves shape
        assert transformed.shape == (5, 2)
    
    def test_scaler_persistence(self, tmp_path):
        """Test saving and loading scaler."""
        from missionguard.preprocessing.transforms import StandardScalerWrapper
        
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        
        scaler = StandardScalerWrapper(["feature1", "feature2"])
        scaler.fit(df)
        
        save_path = tmp_path / "scaler.joblib"
        scaler.save(save_path)
        
        loaded = StandardScalerWrapper.load(save_path)
        
        # Should produce same transform
        original_transform = scaler.transform(df)
        loaded_transform = loaded.transform(df)
        
        pd.testing.assert_frame_equal(original_transform, loaded_transform)
    
    def test_prepare_features_target(self):
        """Test feature/target extraction."""
        from missionguard.preprocessing.transforms import prepare_features_target, get_feature_names
        
        df = pd.DataFrame({
            "segment": [1, 2, 3],
            "anomaly": [0, 1, 0],
            "train": [1, 1, 0],
            "channel": ["A", "A", "A"],
            "sampling": [1, 1, 1],
            "duration": [10, 20, 30],
            "len": [100, 200, 300],
            "mean": [1.0, 2.0, 3.0],
        })
        
        feature_names = get_feature_names(df)
        X, y = prepare_features_target(df, feature_names)
        
        assert list(X.columns) == ["duration", "len", "mean"]
        assert list(y) == [0, 1, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])