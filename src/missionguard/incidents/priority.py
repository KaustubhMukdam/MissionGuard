# src/missionguard/incidents/priority.py
"""Priority scoring for incidents."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .aggregation import Incident


@dataclass
class PriorityScore:
    """Priority score with component breakdown."""
    total_score: float
    components: Dict[str, float]
    weights: Dict[str, float]
    metadata: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Default weights for priority components
DEFAULT_PRIORITY_WEIGHTS = {
    "max_anomaly_score": 0.35,      # Highest single anomaly score
    "mean_anomaly_score": 0.15,     # Average anomaly intensity
    "duration_factor": 0.20,        # Longer incidents = higher priority
    "channel_count_factor": 0.15,   # More channels = higher priority
    "event_count_factor": 0.10,     # More events = higher priority
    "recurrence_factor": 0.05,      # Repeated anomalies
}


class PriorityScorer:
    """Computes priority scores for incidents."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        duration_scale_seconds: float = 3600.0,  # 1 hour reference
        channel_scale: int = 5,  # reference channel count
        event_scale: int = 10,   # reference event count
    ):
        self.weights = weights or DEFAULT_PRIORITY_WEIGHTS.copy()
        self.duration_scale_seconds = duration_scale_seconds
        self.channel_scale = channel_scale
        self.event_scale = event_scale
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def compute_priority(self, incident: Incident) -> PriorityScore:
        """
        Compute priority score for an incident.
        
        Returns:
            PriorityScore with total_score in [0, 1] and component breakdown
        """
        components = {}
        
        # 1. Max anomaly score (0-1 normalized)
        components["max_anomaly_score"] = min(incident.max_anomaly_score, 1.0)
        
        # 2. Mean anomaly score (0-1 normalized)
        components["mean_anomaly_score"] = min(incident.mean_anomaly_score, 1.0)
        
        # 3. Duration factor (logarithmic, normalized to reference)
        dur = incident.duration_seconds
        components["duration_factor"] = min(
            np.log1p(dur) / np.log1p(self.duration_scale_seconds), 1.0
        )
        
        # 4. Channel count factor
        ch_count = incident.channel_count
        components["channel_count_factor"] = min(ch_count / self.channel_scale, 1.0)
        
        # 5. Event count factor
        evt_count = incident.event_count
        components["event_count_factor"] = min(evt_count / self.event_scale, 1.0)
        
        # 6. Recurrence factor (placeholder - could track historical repeats)
        components["recurrence_factor"] = 0.0  # Would need historical data
        
        # Compute weighted total
        total = sum(
            components.get(k, 0) * self.weights.get(k, 0) 
            for k in components
        )
        
        # Metadata for transparency
        metadata = {
            "duration_seconds": incident.duration_seconds,
            "channel_count": incident.channel_count,
            "event_count": incident.event_count,
            "max_anomaly_score": incident.max_anomaly_score,
            "mean_anomaly_score": incident.mean_anomaly_score,
        }
        
        return PriorityScore(
            total_score=total,
            components=components,
            weights=self.weights.copy(),
            metadata=metadata,
        )
    
    def compute_batch(self, incidents: List[Incident]) -> List[PriorityScore]:
        """Compute priority for multiple incidents."""
        return [self.compute_priority(inc) for inc in incidents]
    
    def rank_incidents(self, incidents: List[Incident]) -> List[Incident]:
        """Rank incidents by priority score (highest first)."""
        scored = [(inc, self.compute_priority(inc)) for inc in incidents]
        scored.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Attach priority to incidents
        for inc, score in scored:
            inc.priority_score = score.total_score
            inc.priority_components = score.components
        
        return [inc for inc, _ in scored]


def compute_priority(incident: Incident, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Convenience function to compute priority score for a single incident.
    
    Returns:
        Priority score in [0, 1]
    """
    scorer = PriorityScorer(weights=weights)
    return scorer.compute_priority(incident).total_score


def get_priority_label(score: float) -> str:
    """Convert numeric priority score to label."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.25:
        return "WATCH"
    else:
        return "NOMINAL"


def get_priority_color(score: float) -> str:
    """Get color code for priority score (for UI)."""
    if score >= 0.75:
        return "#ff4444"  # Red
    elif score >= 0.5:
        return "#ffaa00"  # Orange
    elif score >= 0.25:
        return "#ffff00"  # Yellow
    else:
        return "#00ff00"  # Green