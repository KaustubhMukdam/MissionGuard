# src/missionguard/incidents/aggregation.py
"""Temporal incident aggregation from anomaly events."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..detection.events import AnomalyEvent


@dataclass
class Incident:
    """Represents an aggregated operational incident."""
    incident_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    events: List[AnomalyEvent]
    affected_channels: List[str]
    priority_score: Optional[float] = None
    priority_components: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0
    
    @property
    def event_count(self) -> int:
        return len(self.events)
    
    @property
    def max_anomaly_score(self) -> float:
        return max(e.max_score for e in self.events) if self.events else 0.0
    
    @property
    def mean_anomaly_score(self) -> float:
        return np.mean([e.mean_score for e in self.events]) if self.events else 0.0
    
    @property
    def channel_count(self) -> int:
        return len(self.affected_channels)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "duration_minutes": self.duration_minutes,
            "event_count": self.event_count,
            "affected_channels": self.affected_channels,
            "channel_count": self.channel_count,
            "max_anomaly_score": self.max_anomaly_score,
            "mean_anomaly_score": self.mean_anomaly_score,
            "priority_score": self.priority_score,
            "priority_components": self.priority_components,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }


def aggregate_events_to_incidents(
    events: List[AnomalyEvent],
    max_gap_seconds: float = 300.0,  # 5 minutes
    min_events_per_incident: int = 1,
    incident_id_prefix: str = "INC",
) -> List[Incident]:
    """
    Aggregate temporally adjacent anomaly events into incidents.
    
    Args:
        events: List of AnomalyEvent (must be sorted by start_time)
        max_gap_seconds: Maximum time gap between events to merge into same incident
        min_events_per_incident: Minimum events required to form an incident
        incident_id_prefix: Prefix for incident IDs
        
    Returns:
        List of Incident objects
    """
    if not events:
        return []
    
    # Ensure events are sorted by start_time
    sorted_events = sorted(events, key=lambda e: e.start_time)
    
    incidents = []
    current_incident_events = [sorted_events[0]]
    incident_counter = 0
    
    for event in sorted_events[1:]:
        # Check gap from last event in current incident
        last_event = current_incident_events[-1]
        gap_seconds = (event.start_time - last_event.end_time).total_seconds()
        
        if gap_seconds <= max_gap_seconds:
            # Same incident - merge
            current_incident_events.append(event)
        else:
            # Gap too large - finalize current incident
            if len(current_incident_events) >= min_events_per_incident:
                incident = _create_incident(
                    current_incident_events, incident_counter, incident_id_prefix
                )
                incidents.append(incident)
                incident_counter += 1
            current_incident_events = [event]
    
    # Handle last incident
    if len(current_incident_events) >= min_events_per_incident:
        incident = _create_incident(
            current_incident_events, incident_counter, incident_id_prefix
        )
        incidents.append(incident)
    
    return incidents


def _create_incident(
    events: List[AnomalyEvent],
    counter: int,
    prefix: str,
) -> Incident:
    """Create Incident from list of events."""
    start_time = min(e.start_time for e in events)
    end_time = max(e.end_time for e in events)
    affected_channels = sorted(set(e.channel for e in events))
    
    return Incident(
        incident_id=f"{prefix}-{counter:04d}",
        start_time=start_time,
        end_time=end_time,
        events=events,
        affected_channels=affected_channels,
    )


def merge_incidents(
    incidents: List[Incident],
    max_gap_seconds: float = 300.0,
) -> List[Incident]:
    """
    Merge adjacent incidents if they're close in time.
    
    Useful for post-processing after initial aggregation.
    """
    if not incidents:
        return []
    
    sorted_incidents = sorted(incidents, key=lambda i: i.start_time)
    merged = [sorted_incidents[0]]
    
    for incident in sorted_incidents[1:]:
        last = merged[-1]
        gap = (incident.start_time - last.end_time).total_seconds()
        
        if gap <= max_gap_seconds:
            # Merge
            last.events.extend(incident.events)
            last.end_time = max(last.end_time, incident.end_time)
            last.affected_channels = sorted(
                set(last.affected_channels) | set(incident.affected_channels)
            )
        else:
            merged.append(incident)
    
    return merged


def filter_incidents(
    incidents: List[Incident],
    min_duration_seconds: float = 0,
    max_duration_seconds: float = float("inf"),
    min_event_count: int = 1,
    min_priority_score: float = -float("inf"),
    max_priority_score: float = float("inf"),
) -> List[Incident]:
    """Filter incidents by various criteria."""
    filtered = []
    for inc in incidents:
        if inc.duration_seconds < min_duration_seconds:
            continue
        if inc.duration_seconds > max_duration_seconds:
            continue
        if inc.event_count < min_event_count:
            continue
        if inc.priority_score is not None:
            if inc.priority_score < min_priority_score:
                continue
            if inc.priority_score > max_priority_score:
                continue
        filtered.append(inc)
    return filtered


def incidents_to_dataframe(incidents: List[Incident]) -> pd.DataFrame:
    """Convert list of incidents to DataFrame for analysis."""
    if not incidents:
        return pd.DataFrame(columns=[
            "incident_id", "start_time", "end_time", "duration_seconds",
            "event_count", "affected_channels", "channel_count",
            "max_anomaly_score", "mean_anomaly_score", "priority_score"
        ])
    
    rows = []
    for inc in incidents:
        rows.append({
            "incident_id": inc.incident_id,
            "start_time": inc.start_time,
            "end_time": inc.end_time,
            "duration_seconds": inc.duration_seconds,
            "event_count": inc.event_count,
            "affected_channels": ",".join(inc.affected_channels),
            "channel_count": inc.channel_count,
            "max_anomaly_score": inc.max_anomaly_score,
            "mean_anomaly_score": inc.mean_anomaly_score,
            "priority_score": inc.priority_score,
        })
    return pd.DataFrame(rows)