# tests/test_preprocessing.py
"""Tests for preprocessing utilities."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.preprocessing.time_series import (
    sort_by_segment_time,
    extract_segment_windows,
    compute_rolling_features,
    compute_differencing_features,
    detect_gaps,
    align_channels_temporally,
)
from missionguard.preprocessing.transforms import (
    fit_scaler,
    transform_features,
    get_feature_names,
    prepare_features_target,
    StandardScalerWrapper,
    RobustScalerWrapper,
)


class TestTimeSeries:
    """Tests for time series preprocessing."""
    
    def test_sort_by_segment_time(self):
        """Test sorting by segment then timestamp."""
        df = pd.DataFrame({
            "segment": [2, 1, 2, 1],
            "timestamp": pd.to_datetime([
                "2022-01-02", "2022-01-01", "2022-01-03", "2022-01-04"
            ], utc=True),
            "value": [10, 20, 30, 40],
        })
        
        sorted_df = sort_by_segment_time(df)
        
        # Should be segment 1 first, then segment 2
        assert list(sorted_df["segment"]) == [1, 1, 2, 2]
        # Within each segment, timestamps should be sorted
        seg1 = sorted_df[sorted_df["segment"] == 1]["timestamp"]
        assert seg1.is_monotonic_increasing
    
    def test_extract_segment_windows(self):
        """Test extracting a single segment."""
        df = pd.DataFrame({
            "segment": [1, 1, 2, 2],
            "timestamp": pd.to_datetime([
                "2022-01-01", "2022-01-02", "2022-01-01", "2022-01-02"
            ], utc=True),
            "value": [10, 20, 30, 40],
            "anomaly": [0, 1, 0, 0],
        })
        
        seg1 = extract_segment_windows(df, segment_id=1)
        
        assert len(seg1) == 2
        assert list(seg1["value"]) == [10, 20]
        assert seg1["timestamp"].is_monotonic_increasing
    
    def test_compute_rolling_features(self):
        """Test rolling feature computation."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        rolling_df = compute_rolling_features(series, windows=[3, 5], features=["mean", "std"])
        
        assert "roll_3_mean" in rolling_df.columns
        assert "roll_3_std" in rolling_df.columns
        assert "roll_5_mean" in rolling_df.columns
        assert "roll_5_std" in rolling_df.columns
        
        # Check values (min_periods=1 so first values are valid)
        assert rolling_df["roll_3_mean"].iloc[0] == 1.0
        assert rolling_df["roll_3_mean"].iloc[2] == 2.0  # mean of [1,2,3]
    
    def test_compute_differencing_features(self):
        """Test differencing feature computation."""
        series = pd.Series([1, 3, 6, 10, 15])  # Quadratic
        
        diff_df = compute_differencing_features(series, lags=[1, 2])
        
        assert "diff_1" in diff_df.columns
        assert "diff_2" in diff_df.columns
        assert "diff2_1" in diff_df.columns
        assert "diff2_2" in diff_df.columns
        
        # First differences: [NaN, 2, 3, 4, 5]
        # Second differences: [NaN, NaN, 1, 1, 1]
        assert diff_df["diff_1"].iloc[1] == 2
        assert diff_df["diff2_1"].iloc[2] == 1
    
    def test_detect_gaps(self):
        """Test gap detection."""
        df = pd.DataFrame({
            "segment": [1, 1, 1, 2, 2],
            "timestamp": pd.to_datetime([
                "2022-01-01 00:00:00",
                "2022-01-01 00:00:01",  # 1 second gap
                "2022-01-01 00:00:05",  # 4 second gap
                "2022-01-01 01:00:00",
                "2022-01-01 01:00:01",
            ], utc=True),
            "value": [1, 2, 3, 4, 5],
        })
        
        gaps = detect_gaps(df, max_gap_seconds=2.0)
        
        assert len(gaps) == 2
        # Segment 1 has 1 gap > 2 seconds (the 4-second one)
        assert gaps[gaps["segment"] == 1]["num_gaps"].values[0] == 1
        # Segment 2 has no gaps > 2 seconds
        assert gaps[gaps["segment"] == 2]["num_gaps"].values[0] == 0
    
    def test_align_channels_temporally_raises_on_no_overlap(self):
        """Test that align_channels raises error when no time overlap."""
        df = pd.DataFrame({
            "channel": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime([
                "2022-01-01", "2022-01-02",
                "2022-02-01", "2022-02-02",
            ], utc=True),
            "value": [1, 2, 3, 4],
        })
        
        with pytest.raises(ValueError, match="do not have overlapping time ranges"):
            align_channels_temporally(df)


class TestTransforms:
    """Tests for feature transforms."""
    
    def test_fit_scaler_standard(self):
        """Test fitting standard scaler."""
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 0, 1, 1, 1],
        })
        
        scaler = fit_scaler(df, ["feat1", "feat2"], scaler_type="standard")
        
        assert scaler.fitted
        assert isinstance(scaler, StandardScalerWrapper)
    
    def test_fit_scaler_robust(self):
        """Test fitting robust scaler."""
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        
        scaler = fit_scaler(df, ["feat1", "feat2"], scaler_type="robust")
        
        assert scaler.fitted
        assert isinstance(scaler, RobustScalerWrapper)
    
    def test_fit_scaler_invalid_type(self):
        """Test invalid scaler type raises error."""
        df = pd.DataFrame({"feat1": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Unknown scaler_type"):
            fit_scaler(df, ["feat1"], scaler_type="invalid")
    
    def test_transform_features(self):
        """Test transforming features with fitted scaler."""
        from missionguard.preprocessing.transforms import StandardScalerWrapper
        
        train_df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        
        scaler = StandardScalerWrapper(["feat1", "feat2"])
        scaler.fit(train_df)
        
        test_df = pd.DataFrame({
            "feat1": [2.5, 3.5],
            "feat2": [25.0, 35.0],
            "target": [0, 1],
        })
        
        transformed = transform_features(test_df, scaler, ["feat1", "feat2"])
        
        # Target column should be preserved
        assert "target" in transformed.columns
        assert list(transformed["target"]) == [0, 1]
        
        # Features should be scaled
        assert "feat1" in transformed.columns
        assert "feat2" in transformed.columns
    
    def test_get_feature_names(self):
        """Test extracting feature names from dataset."""
        df = pd.DataFrame({
            "segment": [1, 2],
            "anomaly": [0, 1],
            "train": [1, 0],
            "channel": ["A", "A"],
            "sampling": [1, 1],
            "duration": [10, 20],
            "len": [100, 200],
            "mean": [1.0, 2.0],
            "custom_feature": [100, 200],
        })
        
        features = get_feature_names(df)
        
        assert "duration" in features
        assert "len" in features
        assert "mean" in features
        assert "custom_feature" in features
        assert "segment" not in features
        assert "anomaly" not in features
        assert "train" not in features
        assert "channel" not in features
        assert "sampling" not in features
    
    def test_prepare_features_target(self):
        """Test feature/target preparation."""
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
        
        X, y = prepare_features_target(df, ["duration", "len", "mean"])
        
        assert list(X.columns) == ["duration", "len", "mean"]
        assert list(y) == [0, 1, 0]
        assert len(X) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])