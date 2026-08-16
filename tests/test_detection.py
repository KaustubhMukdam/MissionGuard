# tests/test_detection.py
"""Tests for detection module: events, thresholding."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.detection.events import (
    AnomalyEvent,
    scores_to_events,
    merge_events,
    filter_events,
    events_to_dataframe,
    get_events_per_channel,
)
from missionguard.detection.thresholding import (
    ThresholdConfig,
    select_threshold,
    evaluate_threshold,
    find_optimal_threshold,
    evaluate_thresholds_sweep,
    get_false_alarm_rate,
    get_detection_delay,
)


class TestAnomalyEvent:
    """Tests for AnomalyEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating an AnomalyEvent."""
        event = AnomalyEvent(
            channel="CADC0872",
            start_idx=10,
            end_idx=20,
            start_time=pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
            end_time=pd.Timestamp("2022-01-01 00:00:10", tz="UTC"),
            max_score=0.95,
            mean_score=0.85,
            duration_samples=11,
            duration_seconds=10.0,
            segment_ids=[1, 2],
        )
        
        assert event.channel == "CADC0872"
        assert event.duration_minutes == 10.0 / 60
        assert event.duration_hours == 10.0 / 3600
    
    def test_to_dict(self):
        """Test serialization to dict."""
        event = AnomalyEvent(
            channel="CADC0872",
            start_idx=10,
            end_idx=20,
            start_time=pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
            end_time=pd.Timestamp("2022-01-01 00:00:10", tz="UTC"),
            max_score=0.95,
            mean_score=0.85,
            duration_samples=11,
            duration_seconds=10.0,
            segment_ids=[1, 2],
        )
        
        d = event.to_dict()
        assert d["channel"] == "CADC0872"
        assert d["max_score"] == 0.95
        assert d["segment_ids"] == [1, 2]


class TestScoresToEvents:
    """Tests for scores_to_events function."""
    
    def test_basic_event_detection(self):
        """Test detecting a single event."""
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.2, 0.1])
        timestamps = pd.date_range("2022-01-01", periods=7, freq="1s", tz="UTC")
        
        events = scores_to_events(scores, timestamps, "CH1", threshold=0.5)
        
        assert len(events) == 1
        assert events[0].start_idx == 2
        assert events[0].end_idx == 4
        assert events[0].max_score == 0.9
        assert abs(events[0].mean_score - 0.85) < 0.01
        assert events[0].duration_samples == 3
    
    def test_multiple_events(self):
        """Test detecting multiple separate events."""
        scores = np.array([0.8, 0.8, 0.1, 0.1, 0.9, 0.9, 0.1])
        timestamps = pd.date_range("2022-01-01", periods=7, freq="1s", tz="UTC")
        
        events = scores_to_events(scores, timestamps, "CH1", threshold=0.5)
        
        assert len(events) == 2
        assert events[0].start_idx == 0
        assert events[0].end_idx == 1
        assert events[1].start_idx == 4
        assert events[1].end_idx == 5
    
    def test_min_duration_filter(self):
        """Test minimum duration filtering."""
        scores = np.array([0.1, 0.8, 0.1, 0.9, 0.9, 0.1])
        timestamps = pd.date_range("2022-01-01", periods=6, freq="1s", tz="UTC")
        
        # min_duration=2 should filter out single-sample event at idx 1
        events = scores_to_events(scores, timestamps, "CH1", threshold=0.5, min_duration=2)
        
        assert len(events) == 1
        assert events[0].start_idx == 3
        assert events[0].end_idx == 4
    
    def test_event_at_boundaries(self):
        """Test events at start/end of array."""
        scores = np.array([0.8, 0.8, 0.1, 0.1, 0.9, 0.9])
        timestamps = pd.date_range("2022-01-01", periods=6, freq="1s", tz="UTC")
        
        events = scores_to_events(scores, timestamps, "CH1", threshold=0.5)
        
        assert len(events) == 2
        assert events[0].start_idx == 0
        assert events[1].end_idx == 5
    
    def test_segment_ids(self):
        """Test segment ID inclusion."""
        scores = np.array([0.1, 0.8, 0.9, 0.1])
        timestamps = pd.date_range("2022-01-01", periods=4, freq="1s", tz="UTC")
        segments = np.array([1, 2, 2, 3])
        
        events = scores_to_events(scores, timestamps, "CH1", threshold=0.5, segment_ids=segments)
        
        assert len(events) == 1
        assert set(events[0].segment_ids) == {2}


class TestMergeEvents:
    """Tests for merge_events function."""
    
    def test_merge_close_events(self):
        """Test merging events within max_gap."""
        e1 = AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                          0.9, 0.85, 3, 2.0)
        e2 = AnomalyEvent("CH1", 5, 7, pd.Timestamp("2022-01-01 00:00:05", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:07", tz="UTC"),
                          0.8, 0.75, 3, 2.0)
        # Gap is 3 seconds
        
        merged = merge_events([e1, e2], max_gap_seconds=5.0)
        assert len(merged) == 1
        assert merged[0].start_idx == 0
        assert merged[0].end_idx == 7
    
    def test_no_merge_far_events(self):
        """Test not merging events beyond max_gap."""
        e1 = AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                          0.9, 0.85, 3, 2.0)
        e2 = AnomalyEvent("CH1", 10, 12, pd.Timestamp("2022-01-01 00:00:10", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:12", tz="UTC"),
                          0.8, 0.75, 3, 2.0)
        # Gap is 8 seconds
        
        merged = merge_events([e1, e2], max_gap_seconds=5.0)
        assert len(merged) == 2
    
    def test_no_merge_different_channels(self):
        """Test not merging events from different channels."""
        e1 = AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                          0.9, 0.85, 3, 2.0)
        e2 = AnomalyEvent("CH2", 3, 5, pd.Timestamp("2022-01-01 00:00:03", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:05", tz="UTC"),
                          0.8, 0.75, 3, 2.0)
        # Gap is 1 second but different channels
        
        merged = merge_events([e1, e2], max_gap_seconds=5.0)
        assert len(merged) == 2


class TestFilterEvents:
    """Tests for filter_events function."""
    
    def test_filter_by_duration(self):
        """Test filtering by duration."""
        e1 = AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                          0.9, 0.85, 3, 2.0)
        e2 = AnomalyEvent("CH1", 5, 15, pd.Timestamp("2022-01-01 00:00:05", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:15", tz="UTC"),
                          0.8, 0.75, 11, 10.0)
        
        # Keep only events >= 5 seconds
        filtered = filter_events([e1, e2], min_duration_seconds=5.0)
        assert len(filtered) == 1
        assert filtered[0].duration_seconds == 10.0
    
    def test_filter_by_score(self):
        """Test filtering by score."""
        e1 = AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                          0.9, 0.85, 3, 2.0)
        e2 = AnomalyEvent("CH1", 5, 7, pd.Timestamp("2022-01-01 00:00:05", tz="UTC"),
                          pd.Timestamp("2022-01-01 00:00:07", tz="UTC"),
                          0.4, 0.35, 3, 2.0)
        
        filtered = filter_events([e1, e2], min_max_score=0.5)
        assert len(filtered) == 1
        assert filtered[0].max_score == 0.9


class TestThresholding:
    """Tests for threshold selection and evaluation."""
    
    @pytest.fixture
    def sample_scores_labels(self):
        """Create sample scores and labels."""
        np.random.seed(42)
        n_normal = 900
        n_anomaly = 100
        
        normal_scores = np.random.beta(1, 5, n_normal)  # Low scores
        anomaly_scores = np.random.beta(5, 1, n_anomaly)  # High scores
        
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # Shuffle
        idx = np.random.permutation(len(scores))
        return scores[idx], labels[idx]
    
    def test_percentile_threshold(self, sample_scores_labels):
        """Test percentile-based threshold."""
        scores, labels = sample_scores_labels
        
        config = ThresholdConfig(method="percentile", value=95.0)
        threshold = select_threshold(scores, config=config)
        
        expected = np.percentile(scores, 95.0)
        assert abs(threshold - expected) < 1e-6
    
    def test_fixed_threshold(self, sample_scores_labels):
        """Test fixed threshold."""
        scores, labels = sample_scores_labels
        
        config = ThresholdConfig(method="fixed", fixed_threshold=0.5)
        threshold = select_threshold(scores, config=config)
        
        assert threshold == 0.5
    
    def test_f1_optimal_threshold(self, sample_scores_labels):
        """Test F1-optimal threshold."""
        scores, labels = sample_scores_labels
        
        config = ThresholdConfig(method="f1_optimal")
        threshold = select_threshold(scores, labels, config=config)
        
        assert 0 < threshold < 1
    
    def test_evaluate_threshold(self, sample_scores_labels):
        """Test threshold evaluation."""
        scores, labels = sample_scores_labels
        
        metrics = evaluate_threshold(scores, labels, threshold=0.5)
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "tp" in metrics
        assert "fp" in metrics
        assert "tn" in metrics
        assert "fn" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
    
    def test_find_optimal_threshold(self, sample_scores_labels):
        """Test finding optimal threshold."""
        scores, labels = sample_scores_labels
        
        threshold, metrics = find_optimal_threshold(scores, labels, metric="f1")
        
        assert 0 < threshold < 1
        assert metrics["f1"] >= 0
    
    def test_evaluate_thresholds_sweep(self, sample_scores_labels):
        """Test sweeping multiple thresholds."""
        scores, labels = sample_scores_labels
        
        df = evaluate_thresholds_sweep(scores, labels, n_thresholds=10)
        
        assert len(df) == 10
        assert "threshold" in df.columns
        assert "f1" in df.columns
    
    def test_false_alarm_rate(self, sample_scores_labels):
        """Test false alarm rate computation."""
        scores, labels = sample_scores_labels
        
        far = get_false_alarm_rate(scores, labels, threshold=0.5, time_per_sample=1.0)
        
        assert "false_positives" in far
        assert "false_alarms_per_hour" in far
        assert "false_alarm_rate" in far
        assert "mtbfa_hours" in far
    
    def test_detection_delay(self, sample_scores_labels):
        """Test detection delay computation."""
        scores, labels = sample_scores_labels
        
        delay = get_detection_delay(scores, labels, threshold=0.5, time_per_sample=1.0)
        
        assert "detected_events" in delay
        assert "mean_delay_seconds" in delay
        assert "median_delay_seconds" in delay
        assert "max_delay_seconds" in delay


class TestEventsToDataFrame:
    """Tests for events_to_dataframe."""
    
    def test_empty_list(self):
        """Test empty event list."""
        df = events_to_dataframe([])
        assert len(df) == 0
        assert list(df.columns) == [
            "channel", "start_idx", "end_idx", "start_time", "end_time",
            "max_score", "mean_score", "duration_samples", "duration_seconds", "segment_ids"
        ]
    
    def test_multiple_events(self):
        """Test converting multiple events."""
        events = [
            AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01", tz="UTC"),
                         pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                         0.9, 0.85, 3, 2.0),
            AnomalyEvent("CH1", 5, 7, pd.Timestamp("2022-01-01 00:00:05", tz="UTC"),
                         pd.Timestamp("2022-01-01 00:00:07", tz="UTC"),
                         0.8, 0.75, 3, 2.0),
        ]
        
        df = events_to_dataframe(events)
        assert len(df) == 2
        assert df.iloc[0]["max_score"] == 0.9
        assert df.iloc[1]["max_score"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])