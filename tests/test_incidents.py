# tests/test_incidents.py
"""Tests for incident engine: aggregation, priority, evidence."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from missionguard.incidents.aggregation import (
    Incident,
    aggregate_events_to_incidents,
    merge_incidents,
    filter_incidents,
    incidents_to_dataframe,
)
from missionguard.incidents.priority import (
    PriorityScorer,
    PriorityScore,
    compute_priority,
    DEFAULT_PRIORITY_WEIGHTS,
    get_priority_label,
    get_priority_color,
)
from missionguard.incidents.evidence import (
    EvidencePacket,
    build_evidence_packet,
    serialize_evidence_packet,
    validate_evidence_packet,
)
from missionguard.detection.events import AnomalyEvent


# Module-level fixture for sample events
@pytest.fixture
def sample_events():
    """Create sample anomaly events."""
    base_time = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
    return [
        AnomalyEvent(
            channel="CH1",
            start_idx=0,
            end_idx=2,
            start_time=base_time,
            end_time=base_time + timedelta(seconds=2),
            max_score=0.9,
            mean_score=0.85,
            duration_samples=3,
            duration_seconds=2.0,
        ),
        AnomalyEvent(
            channel="CH1",
            start_idx=5,
            end_idx=7,
            start_time=base_time + timedelta(seconds=5),
            end_time=base_time + timedelta(seconds=7),
            max_score=0.8,
            mean_score=0.75,
            duration_samples=3,
            duration_seconds=2.0,
        ),
        AnomalyEvent(
            channel="CH2",
            start_idx=10,
            end_idx=12,
            start_time=base_time + timedelta(seconds=10),
            end_time=base_time + timedelta(seconds=12),
            max_score=0.85,
            mean_score=0.8,
            duration_samples=3,
            duration_seconds=2.0,
        ),
        # Far away event - should be separate incident
        AnomalyEvent(
            channel="CH1",
            start_idx=100,
            end_idx=102,
            start_time=base_time + timedelta(seconds=100),
            end_time=base_time + timedelta(seconds=102),
            max_score=0.95,
            mean_score=0.9,
            duration_samples=3,
            duration_seconds=2.0,
        ),
    ]


class TestIncidentAggregation:
    """Tests for event-to-incident aggregation."""

    def test_basic_aggregation(self, sample_events):
        """Test basic temporal aggregation."""
        incidents = aggregate_events_to_incidents(
            sample_events,
            max_gap_seconds=10.0,
            min_events_per_incident=1,
        )

        assert len(incidents) == 2  # First 3 events merge, last is separate

        # First incident: 3 events, 2 channels
        inc1 = incidents[0]
        assert inc1.event_count == 3
        assert inc1.channel_count == 2
        assert inc1.affected_channels == ["CH1", "CH2"]
        assert inc1.max_anomaly_score == 0.9

        # Second incident: 1 event
        inc2 = incidents[1]
        assert inc2.event_count == 1
        assert inc2.channel_count == 1

    def test_min_events_filter(self, sample_events):
        """Test minimum events per incident filter."""
        incidents = aggregate_events_to_incidents(
            sample_events,
            max_gap_seconds=10.0,
            min_events_per_incident=2,  # Require at least 2 events
        )

        assert len(incidents) == 1  # Only first group has 3 events
        assert incidents[0].event_count == 3

    def test_incident_properties(self, sample_events):
        """Test computed incident properties."""
        incidents = aggregate_events_to_incidents(sample_events, max_gap_seconds=10.0)
        inc = incidents[0]

        assert inc.duration_seconds > 0
        assert inc.duration_minutes > 0
        assert inc.event_count == 3
        assert inc.max_anomaly_score == 0.9
        # Mean of 0.85, 0.75, 0.80 = 0.80
        assert abs(inc.mean_anomaly_score - 0.80) < 0.01

    def test_incident_to_dict(self, sample_events):
        """Test incident serialization."""
        incidents = aggregate_events_to_incidents(sample_events, max_gap_seconds=10.0)
        d = incidents[0].to_dict()

        assert "incident_id" in d
        assert "start_time" in d
        assert "end_time" in d
        assert "events" in d
        assert len(d["events"]) == 3


class TestMergeIncidents:
    """Tests for merging adjacent incidents."""

    def test_merge_close_incidents(self):
        """Test merging incidents within gap threshold."""
        base = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")

        inc1 = Incident(
            incident_id="INC-001",
            start_time=base,
            end_time=base + timedelta(seconds=10),
            events=[],
            affected_channels=["CH1"],
        )
        inc2 = Incident(
            incident_id="INC-002",
            start_time=base + timedelta(seconds=30),  # 20s gap
            end_time=base + timedelta(seconds=40),
            events=[],
            affected_channels=["CH2"],
        )

        merged = merge_incidents([inc1, inc2], max_gap_seconds=60.0)
        assert len(merged) == 1
        assert merged[0].channel_count == 2
        assert "CH1" in merged[0].affected_channels
        assert "CH2" in merged[0].affected_channels

    def test_no_merge_far_incidents(self):
        """Test not merging incidents beyond gap threshold."""
        base = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")

        inc1 = Incident(
            incident_id="INC-001",
            start_time=base,
            end_time=base + timedelta(seconds=10),
            events=[],
            affected_channels=["CH1"],
        )
        inc2 = Incident(
            incident_id="INC-002",
            start_time=base + timedelta(seconds=100),  # 90s gap
            end_time=base + timedelta(seconds=110),
            events=[],
            affected_channels=["CH2"],
        )

        merged = merge_incidents([inc1, inc2], max_gap_seconds=60.0)
        assert len(merged) == 2


class TestFilterIncidents:
    """Tests for filtering incidents."""

    @pytest.fixture
    def sample_incidents(self):
        base = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
        return [
            Incident("INC-001", base, base + timedelta(seconds=10),
                     [AnomalyEvent("CH1", 0, 1, base, base + timedelta(seconds=10), 0.9, 0.8, 1, 10.0)],
                     ["CH1"], priority_score=0.8),
            Incident("INC-002", base + timedelta(minutes=5), base + timedelta(minutes=6),
                     [AnomalyEvent("CH2", 0, 1, base + timedelta(minutes=5), base + timedelta(minutes=6), 0.5, 0.4, 1, 10.0)],
                     ["CH2"], priority_score=0.3),
            Incident("INC-003", base + timedelta(minutes=10), base + timedelta(minutes=11),
                     [AnomalyEvent("CH1", 0, 1, base + timedelta(minutes=10), base + timedelta(minutes=11), 0.7, 0.6, 1, 10.0)],
                     ["CH1"], priority_score=0.6),
        ]

    def test_filter_by_priority(self, sample_incidents):
        """Test filtering by priority score."""
        filtered = filter_incidents(sample_incidents, min_priority_score=0.5)
        assert len(filtered) == 2
        assert all(i.priority_score >= 0.5 for i in filtered)

    def test_filter_by_duration(self, sample_incidents):
        """Test filtering by duration."""
        # INC-001: 10 seconds, INC-002: 60 seconds, INC-003: 60 seconds
        # With min_duration_seconds=20, only INC-001 should be filtered out
        filtered = filter_incidents(sample_incidents, min_duration_seconds=20)
        assert len(filtered) == 2  # INC-002 and INC-003 pass (60s each)
        assert all(i.duration_seconds >= 20 for i in filtered)


class TestIncidentsDataFrame:
    """Tests for DataFrame conversion."""

    def test_incidents_to_dataframe(self, sample_events):
        """Test conversion to DataFrame."""
        incidents = aggregate_events_to_incidents(sample_events, max_gap_seconds=10.0)
        df = incidents_to_dataframe(incidents)

        assert len(df) == 2
        assert "incident_id" in df.columns
        assert "duration_seconds" in df.columns
        assert "affected_channels" in df.columns


class TestPriorityScoring:
    """Tests for priority scoring."""

    @pytest.fixture
    def sample_incident(self):
        base = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
        events = [
            AnomalyEvent("CH1", 0, 1, base, base + timedelta(seconds=10), 0.95, 0.8, 1, 10.0),
            AnomalyEvent("CH2", 0, 1, base + timedelta(seconds=100), base + timedelta(seconds=110), 0.9, 0.85, 1, 10.0),
            AnomalyEvent("CH3", 0, 1, base + timedelta(seconds=200), base + timedelta(seconds=210), 0.92, 0.88, 1, 10.0),
        ]
        return Incident(
            incident_id="INC-001",
            start_time=base,
            end_time=base + timedelta(seconds=300),  # 5 minutes
            events=events,
            affected_channels=["CH1", "CH2", "CH3"],
        )

    def test_priority_scoring(self, sample_incident):
        """Test basic priority computation."""
        scorer = PriorityScorer()
        score = scorer.compute_priority(sample_incident)

        assert isinstance(score, PriorityScore)
        assert 0 <= score.total_score <= 1
        assert "max_anomaly_score" in score.components
        assert "duration_factor" in score.components

    def test_priority_label(self):
        """Test priority label mapping."""
        assert get_priority_label(0.9) == "CRITICAL"
        assert get_priority_label(0.6) == "HIGH"
        assert get_priority_label(0.3) == "WATCH"
        assert get_priority_label(0.1) == "NOMINAL"

    def test_priority_color(self):
        """Test priority color mapping."""
        assert get_priority_color(0.9) == "#ff4444"
        assert get_priority_color(0.6) == "#ffaa00"
        assert get_priority_color(0.3) == "#ffff00"
        assert get_priority_color(0.1) == "#00ff00"

    def test_custom_weights(self, sample_incident):
        """Test custom weight configuration."""
        custom_weights = {
            "max_anomaly_score": 0.5,
            "duration_factor": 0.5,
            "mean_anomaly_score": 0.0,
            "channel_count_factor": 0.0,
            "event_count_factor": 0.0,
            "recurrence_factor": 0.0,
        }
        scorer = PriorityScorer(weights=custom_weights)
        score = scorer.compute_priority(sample_incident)

        # With only max_anomaly_score=0.95 and duration_factor contributing
        assert score.total_score > 0

    def test_rank_incidents(self):
        """Test incident ranking."""
        base = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")

        def make_incident(inc_id, max_score, mean_score):
            events = [AnomalyEvent("CH1", 0, 1, base, base + timedelta(seconds=10),
                                   max_score, mean_score, 1, 10.0)]
            return Incident(inc_id, base, base + timedelta(seconds=10), events, ["CH1"])

        incidents = [
            make_incident("INC-001", 0.5, 0.4),
            make_incident("INC-002", 0.9, 0.8),
            make_incident("INC-003", 0.7, 0.6),
        ]

        scorer = PriorityScorer()
        ranked = scorer.rank_incidents(incidents)

        # INC-002 should be first (highest anomaly scores)
        assert ranked[0].incident_id == "INC-002"
        assert ranked[1].incident_id == "INC-003"
        assert ranked[2].incident_id == "INC-001"


class TestEvidencePacket:
    """Tests for evidence packet construction."""

    @pytest.fixture
    def sample_incident(self):
        base = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
        events = [
            AnomalyEvent("CH1", 0, 2, pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                         pd.Timestamp("2022-01-01 00:00:02", tz="UTC"),
                         0.9, 0.85, 3, 2.0),
        ]
        return Incident("INC-001", pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
                        pd.Timestamp("2022-01-01 00:00:10", tz="UTC"),
                        events, ["CH1"])

    def test_build_evidence_packet(self, sample_incident):
        """Test building evidence packet."""
        from missionguard.incidents.priority import PriorityScorer

        scorer = PriorityScorer()
        priority = scorer.compute_priority(sample_incident)

        packet = build_evidence_packet(
            sample_incident,
            priority=priority,
            model_info={
                "name": "IsolationForestDetector",
                "version": "1.0.0",
                "experiment_id": "EXP-005",
                "threshold": 0.15,
                "score_normalization": "minmax",
            },
            evaluation_metrics={
                "precision": 0.30,
                "recall": 0.80,
                "f1": 0.44,
                "roc_auc": 0.64,
                "false_alarms_per_hour": 1500,
                "mean_detection_delay_seconds": 0.1,
            },
        )

        assert isinstance(packet, EvidencePacket)
        assert packet.incident_id == "INC-001"
        assert packet.max_anomaly_score == 0.9
        assert packet.priority_score is not None
        assert packet.model_name == "IsolationForestDetector"
        assert packet.evaluation_f1 == 0.44

    def test_evidence_packet_serialization(self, sample_incident):
        """Test JSON serialization."""
        packet = EvidencePacket(
            incident_id="TEST-001",
            start_time="2022-01-01T00:00:00+00:00",
            end_time="2022-01-01T00:00:10+00:00",
            duration_seconds=10.0,
            affected_channels=["CH1"],
            channel_count=1,
            anomaly_events=[],
            max_anomaly_score=0.9,
            mean_anomaly_score=0.8,
            event_count=1,
        )

        json_str = packet.to_json()
        assert "TEST-001" in json_str
        assert "max_anomaly_score" in json_str

    def test_evidence_packet_validation(self):
        """Test validation warnings."""
        # Valid packet
        packet = EvidencePacket(
            incident_id="TEST-001",
            start_time="2022-01-01T00:00:00+00:00",
            end_time="2022-01-01T00:00:10+00:00",
            duration_seconds=10.0,
            affected_channels=["CH1"],
            channel_count=1,
            anomaly_events=[{"max_score": 0.9}],
            max_anomaly_score=0.9,
            mean_anomaly_score=0.8,
            event_count=1,
            priority_score=0.5,
            model_name="TestModel",
            evaluation_f1=0.5,
        )

        warnings = validate_evidence_packet(packet)
        # Should have no critical warnings
        critical = [w for w in warnings if "Missing" in w]
        assert len(critical) == 0

    def test_validation_catches_missing_fields(self):
        """Test validation catches missing required fields."""
        packet = EvidencePacket(
            incident_id="",  # Missing
            start_time="2022-01-01T00:00:00+00:00",
            end_time="2022-01-01T00:00:10+00:00",
            duration_seconds=10.0,
            affected_channels=[],
            channel_count=0,
            anomaly_events=[],
            max_anomaly_score=None,
            mean_anomaly_score=0.8,
            event_count=0,
        )

        warnings = validate_evidence_packet(packet)
        assert any("Missing incident_id" in w for w in warnings)
        assert any("Missing model_name" in w for w in warnings)
        assert any("Missing evaluation metrics" in w for w in warnings)


class TestEvidenceSerialization:
    """Tests for evidence packet serialization."""

    def test_json_serialization(self):
        """Test JSON serialization."""
        packet = EvidencePacket(
            incident_id="TEST-001",
            start_time="2022-01-01T00:00:00+00:00",
            end_time="2022-01-01T00:00:10+00:00",
            duration_seconds=10.0,
            affected_channels=["CH1"],
            channel_count=1,
            anomaly_events=[],
            max_anomaly_score=0.9,
            mean_anomaly_score=0.8,
            event_count=1,
        )

        json_str = serialize_evidence_packet(packet, format="json")
        assert "TEST-001" in json_str

    def test_yaml_serialization(self):
        """Test YAML serialization works if PyYAML is installed."""
        packet = EvidencePacket(
            incident_id="TEST-001",
            start_time="2022-01-01T00:00:00+00:00",
            end_time="2022-01-01T00:00:10+00:00",
            duration_seconds=10.0,
            affected_channels=["CH1"],
            channel_count=1,
            anomaly_events=[],
            max_anomaly_score=0.9,
            mean_anomaly_score=0.8,
            event_count=1,
        )

        try:
            yaml_str = serialize_evidence_packet(packet, format="yaml")
            assert "TEST-001" in yaml_str
        except ValueError as e:
            if "PyYAML required" in str(e):
                pytest.skip("PyYAML not installed")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])