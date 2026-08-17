# src/missionguard/incidents/evidence.py
"""Evidence packet construction for incidents."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import pandas as pd
import numpy as np

from .aggregation import Incident
from .priority import PriorityScore, get_priority_label
from ..detection.events import AnomalyEvent
from ..models.base import BaseAnomalyDetector


@dataclass
class EvidencePacket:
    """
    Structured evidence packet for an incident.
    
    This is the contract between the analytical system and the LLM explanation layer.
    Every field must be traceable to underlying data.
    """
    # Incident identification
    incident_id: str
    start_time: str  # ISO format
    end_time: str
    duration_seconds: float
    
    # Affected systems
    affected_channels: List[str]
    channel_count: int
    
    # Anomaly evidence
    anomaly_events: List[Dict[str, Any]]  # List of event dicts
    max_anomaly_score: float
    mean_anomaly_score: float
    event_count: int
    
    # Priority
    priority_score: Optional[float] = None
    priority_components: Optional[Dict[str, float]] = None
    priority_weights: Optional[Dict[str, float]] = None
    priority_label: Optional[str] = None
    
    # Model information
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    experiment_id: Optional[str] = None
    threshold_used: Optional[float] = None
    score_normalization: Optional[str] = None
    
    # Evaluation context
    evaluation_precision: Optional[float] = None
    evaluation_recall: Optional[float] = None
    evaluation_f1: Optional[float] = None
    evaluation_roc_auc: Optional[float] = None
    false_alarms_per_hour: Optional[float] = None
    mean_detection_delay_seconds: Optional[float] = None
    
    # Data quality
    data_gaps: Optional[bool] = None
    missing_channels: List[str] = field(default_factory=list)
    sampling_rates: List[int] = field(default_factory=list)
    
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generated_by: str = "MissionGuard Incident Engine"
    schema_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, path: str) -> None:
        """Save evidence packet to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())


def build_evidence_packet(
    incident: Incident,
    priority: Optional[PriorityScore] = None,
    model_info: Optional[Dict[str, Any]] = None,
    evaluation_metrics: Optional[Dict[str, float]] = None,
    data_quality: Optional[Dict[str, Any]] = None,
) -> EvidencePacket:
    """
    Build structured evidence packet from incident and context.
    
    Args:
        incident: The Incident object
        priority: PriorityScore (optional)
        model_info: Dict with model_name, version, experiment_id, threshold, score_normalization
        evaluation_metrics: Dict with precision, recall, f1, roc_auc, false_alarms_per_hour, mean_detection_delay_seconds
        data_quality: Dict with gaps, missing_channels, sampling_rates
        
    Returns:
        EvidencePacket ready for LLM briefing
    """
    # Convert events to serializable dicts
    event_dicts = [e.to_dict() for e in incident.events]
    
    # Priority info
    priority_score = None
    priority_components = None
    priority_weights = None
    priority_label = None
    
    if priority is not None:
        priority_score = priority.total_score
        priority_components = priority.components
        priority_weights = priority.weights
        priority_label = get_priority_label(priority.total_score)
    
    # Model info
    model_name = model_version = experiment_id = None
    threshold_used = score_normalization = None
    if model_info:
        model_name = model_info.get("name")
        model_version = model_info.get("version")
        experiment_id = model_info.get("experiment_id")
        threshold_used = model_info.get("threshold")
        score_normalization = model_info.get("score_normalization")
    
    # Evaluation metrics
    eval_precision = eval_recall = eval_f1 = eval_roc_auc = None
    false_alarms_per_hour = mean_delay = None
    if evaluation_metrics:
        eval_precision = evaluation_metrics.get("precision")
        eval_recall = evaluation_metrics.get("recall")
        eval_f1 = evaluation_metrics.get("f1")
        eval_roc_auc = evaluation_metrics.get("roc_auc")
        false_alarms_per_hour = evaluation_metrics.get("false_alarms_per_hour")
        mean_delay = evaluation_metrics.get("mean_detection_delay_seconds")
    
    # Data quality
    gaps = missing = sampling = None
    if data_quality:
        gaps = data_quality.get("gaps")
        missing = data_quality.get("missing_channels", [])
        sampling = data_quality.get("sampling_rates", [])
    
    return EvidencePacket(
        incident_id=incident.incident_id,
        start_time=incident.start_time.isoformat(),
        end_time=incident.end_time.isoformat(),
        duration_seconds=incident.duration_seconds,
        affected_channels=incident.affected_channels,
        channel_count=incident.channel_count,
        anomaly_events=event_dicts,
        max_anomaly_score=incident.max_anomaly_score,
        mean_anomaly_score=incident.mean_anomaly_score,
        event_count=incident.event_count,
        priority_score=priority_score,
        priority_components=priority_components,
        priority_weights=priority_weights,
        priority_label=priority_label,
        model_name=model_name,
        model_version=model_version,
        experiment_id=experiment_id,
        threshold_used=threshold_used,
        score_normalization=score_normalization,
        evaluation_precision=eval_precision,
        evaluation_recall=eval_recall,
        evaluation_f1=eval_f1,
        evaluation_roc_auc=eval_roc_auc,
        false_alarms_per_hour=false_alarms_per_hour,
        mean_detection_delay_seconds=mean_delay,
        data_gaps=gaps,
        missing_channels=missing or [],
        sampling_rates=sampling or [],
    )


def build_evidence_packet_from_raw(
    incident_id: str,
    start_time: datetime,
    end_time: datetime,
    events: List[AnomalyEvent],
    affected_channels: List[str],
    max_score: float,
    mean_score: float,
    **kwargs
) -> EvidencePacket:
    """Build evidence packet from raw components (for testing/manual creation)."""
    incident = Incident(
        incident_id=incident_id,
        start_time=start_time,
        end_time=end_time,
        events=events,
        affected_channels=affected_channels,
    )
    return build_evidence_packet(incident, **kwargs)


def serialize_evidence_packet(packet: EvidencePacket, format: str = "json") -> str:
    """
    Serialize evidence packet to string.
    
    Args:
        packet: EvidencePacket
        format: "json" or "yaml"
        
    Returns:
        Serialized string
    """
    if format == "json":
        return packet.to_json()
    elif format == "yaml":
        try:
            import yaml
            return yaml.dump(packet.to_dict(), default_flow_style=False)
        except ImportError:
            raise ValueError("PyYAML required for YAML format")
    else:
        raise ValueError(f"Unknown format: {format}")


# Template for LLM prompt (grounded briefing)
LLM_BRIEFING_TEMPLATE = """
You are a spacecraft operations analyst. Generate a concise operator briefing for the incident below.

INCIDENT EVIDENCE:
{evidence_json}

CONSTRAINTS:
1. Use ONLY the evidence provided. Do NOT invent telemetry values, channels, or timestamps.
2. Do NOT claim causal root cause or physical diagnosis.
3. Distinguish observations from hypotheses.
4. State uncertainty explicitly.
5. Provide 2-3 actionable investigation suggestions.
6. End with: "⚠ Decision Support Only. Not for autonomous diagnosis."

STRUCTURE YOUR RESPONSE AS:
**Summary** (1-2 sentences): What happened?
**Why Flagged** (1-2 sentences): Which evidence triggered the alert?
**Investigation Suggestions** (2-3 bullets): What should the operator check?
"""


def build_llm_prompt(evidence_packet: EvidencePacket) -> str:
    """Build grounded prompt for LLM briefing."""
    evidence_json = json.dumps(evidence_packet.to_dict(), indent=2, default=str)
    return LLM_BRIEFING_TEMPLATE.format(evidence_json=evidence_json)


def validate_evidence_packet(packet: EvidencePacket) -> List[str]:
    """
    Validate evidence packet for completeness.
    
    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []
    
    if not packet.incident_id:
        warnings.append("Missing incident_id")
    if not packet.anomaly_events:
        warnings.append("No anomaly events in packet")
    if packet.max_anomaly_score is None:
        warnings.append("Missing max_anomaly_score")
    if packet.priority_score is None:
        warnings.append("Missing priority_score")
    if packet.model_name is None:
        warnings.append("Missing model_name")
    if packet.evaluation_f1 is None:
        warnings.append("Missing evaluation metrics (f1, precision, recall)")
    
    # Check for NaN values
    import math
    for key, value in packet.to_dict().items():
        if isinstance(value, float) and math.isnan(value):
            warnings.append(f"NaN value in field: {key}")
    
    return warnings