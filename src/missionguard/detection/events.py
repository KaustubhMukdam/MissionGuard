# src/missionguard/detection/events.py
"""Anomaly event representation and score-to-event conversion."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class AnomalyEvent:
    """Represents a contiguous anomaly event."""
    channel: str
    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    max_score: float
    mean_score: float
    duration_samples: int
    duration_seconds: float
    segment_ids: List[int] = field(default_factory=list)
    
    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0
    
    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0
    
    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_time": self.start_time.isoformat() if pd.notna(self.start_time) else None,
            "end_time": self.end_time.isoformat() if pd.notna(self.end_time) else None,
            "max_score": float(self.max_score),
            "mean_score": float(self.mean_score),
            "duration_samples": int(self.duration_samples),
            "duration_seconds": float(self.duration_seconds),
            "segment_ids": self.segment_ids,
        }


def scores_to_events(
    scores: np.ndarray,
    timestamps: pd.Series,
    channel: str,
    threshold: float,
    min_duration: int = 1,
    segment_ids: Optional[np.ndarray] = None,
) -> List[AnomalyEvent]:
    """
    Convert anomaly scores to discrete events.
    
    Args:
        scores: Anomaly scores (same length as timestamps)
        timestamps: Timestamps for each score
        channel: Channel identifier
        threshold: Score threshold for anomaly
        min_duration: Minimum consecutive samples to form an event
        segment_ids: Optional segment IDs for each sample
        
    Returns:
        List of AnomalyEvent objects
    """
    if len(scores) != len(timestamps):
        raise ValueError("scores and timestamps must have same length")
    
    # Binary anomaly mask
    is_anomaly = scores >= threshold
    
    # Find contiguous regions
    events = []
    in_event = False
    event_start = 0
    
    for i, anomalous in enumerate(is_anomaly):
        if anomalous and not in_event:
            # Start new event
            in_event = True
            event_start = i
        elif not anomalous and in_event:
            # End event
            in_event = False
            event_end = i - 1
            
            if event_end - event_start + 1 >= min_duration:
                event = _create_event(
                    scores, timestamps, channel, event_start, event_end, segment_ids
                )
                events.append(event)
    
    # Handle event that extends to end
    if in_event:
        event_end = len(scores) - 1
        if event_end - event_start + 1 >= min_duration:
            event = _create_event(
                scores, timestamps, channel, event_start, event_end, segment_ids
            )
            events.append(event)
    
    return events


def _create_event(
    scores: np.ndarray,
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    channel: str,
    start_idx: int,
    end_idx: int,
    segment_ids: Optional[np.ndarray] = None,
) -> AnomalyEvent:
    """Create AnomalyEvent from indices."""
    event_scores = scores[start_idx:end_idx+1]
    # Handle both Series and DatetimeIndex
    event_times = timestamps[start_idx:end_idx+1]
    
    duration_seconds = (event_times[-1] - event_times[0]).total_seconds()
    if duration_seconds < 0:
        duration_seconds = 0
    
    seg_ids = []
    if segment_ids is not None:
        seg_ids = segment_ids[start_idx:end_idx+1].tolist()
        # Unique segment IDs
        seg_ids = sorted(set(seg_ids))
    
    return AnomalyEvent(
        channel=channel,
        start_idx=start_idx,
        end_idx=end_idx,
        start_time=event_times[0],
        end_time=event_times[-1],
        max_score=float(event_scores.max()),
        mean_score=float(event_scores.mean()),
        duration_samples=end_idx - start_idx + 1,
        duration_seconds=duration_seconds,
        segment_ids=seg_ids,
    )


def merge_events(
    events: List[AnomalyEvent],
    max_gap_seconds: float = 300.0,  # 5 minutes
) -> List[AnomalyEvent]:
    """
    Merge nearby events from the same channel.
    
    Args:
        events: List of AnomalyEvent (must be sorted by start_time)
        max_gap_seconds: Maximum gap between events to merge
        
    Returns:
        Merged list of events
    """
    if not events:
        return []
    
    # Sort by start time
    sorted_events = sorted(events, key=lambda e: e.start_time)
    
    merged = [sorted_events[0]]
    
    for event in sorted_events[1:]:
        last = merged[-1]
        
        # Check if same channel and gap is small enough
        if (event.channel == last.channel and
            (event.start_time - last.end_time).total_seconds() <= max_gap_seconds):
            # Merge: extend last event
            last.end_idx = event.end_idx
            last.end_time = event.end_time
            last.max_score = max(last.max_score, event.max_score)
            # Recalculate mean score (weighted by duration)
            total_dur = last.duration_samples + event.duration_samples
            last.mean_score = (
                last.mean_score * last.duration_samples + 
                event.mean_score * event.duration_samples
            ) / total_dur
            last.duration_samples = total_dur
            last.duration_seconds = (last.end_time - last.start_time).total_seconds()
            last.segment_ids = sorted(set(last.segment_ids + event.segment_ids))
        else:
            merged.append(event)
    
    return merged


def filter_events(
    events: List[AnomalyEvent],
    min_duration_seconds: float = 0,
    max_duration_seconds: float = float("inf"),
    min_max_score: float = -float("inf"),
    min_mean_score: float = -float("inf"),
) -> List[AnomalyEvent]:
    """
    Filter events by duration and score criteria.
    
    Args:
        events: List of AnomalyEvent
        min_duration_seconds: Minimum event duration
        max_duration_seconds: Maximum event duration
        min_max_score: Minimum max score
        min_mean_score: Minimum mean score
        
    Returns:
        Filtered list of events
    """
    filtered = []
    for event in events:
        if event.duration_seconds < min_duration_seconds:
            continue
        if event.duration_seconds > max_duration_seconds:
            continue
        if event.max_score < min_max_score:
            continue
        if event.mean_score < min_mean_score:
            continue
        filtered.append(event)
    return filtered


def events_to_dataframe(events: List[AnomalyEvent]) -> pd.DataFrame:
    """Convert list of events to DataFrame."""
    if not events:
        return pd.DataFrame(columns=[
            "channel", "start_idx", "end_idx", "start_time", "end_time",
            "max_score", "mean_score", "duration_samples", "duration_seconds", "segment_ids"
        ])
    
    data = [e.to_dict() for e in events]
    return pd.DataFrame(data)


def get_events_per_channel(
    segments_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    timestamp_col: str = "timestamp",
    channel_col: str = "channel",
    segment_col: str = "segment",
    min_duration: int = 1,
) -> dict:
    """
    Extract events for each channel from scored segments.
    
    Args:
        segments_df: Raw telemetry DataFrame with scores added
        scores: Anomaly scores (aligned with segments_df)
        threshold: Anomaly threshold
        timestamp_col: Timestamp column name
        channel_col: Channel column name
        segment_col: Segment column name
        min_duration: Minimum event duration in samples
        
    Returns:
        Dict mapping channel -> List[AnomalyEvent]
    """
    segments_df = segments_df.copy()
    segments_df["anomaly_score"] = scores
    
    events_by_channel = {}
    
    for channel in segments_df[channel_col].unique():
        ch_data = segments_df[segments_df[channel_col] == channel].copy()
        ch_data = ch_data.sort_values(timestamp_col)
        
        ch_scores = ch_data["anomaly_score"].values
        ch_times = ch_data[timestamp_col]
        ch_segments = ch_data[segment_col].values if segment_col in ch_data.columns else None
        
        events = scores_to_events(
            ch_scores, ch_times, channel, threshold,
            min_duration=min_duration, segment_ids=ch_segments
        )
        
        if events:
            events_by_channel[channel] = events
    
    return events_by_channel