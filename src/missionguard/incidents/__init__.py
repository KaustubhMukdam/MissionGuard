# src/missionguard/incidents/__init__.py
"""Incidents module: temporal aggregation, priority scoring, evidence packets."""

from .aggregation import (
    Incident,
    aggregate_events_to_incidents,
    merge_incidents,
)

from .priority import (
    PriorityScorer,
    PriorityScore,
    compute_priority,
    DEFAULT_PRIORITY_WEIGHTS,
)

from .evidence import (
    EvidencePacket,
    build_evidence_packet,
    serialize_evidence_packet,
)

__all__ = [
    "Incident",
    "aggregate_events_to_incidents",
    "merge_incidents",
    "PriorityScorer",
    "PriorityScore",
    "compute_priority",
    "DEFAULT_PRIORITY_WEIGHTS",
    "EvidencePacket",
    "build_evidence_packet",
    "serialize_evidence_packet",
]