# src/missionguard/detection/__init__.py
"""Detection module: score-to-event conversion and event processing."""

from .events import (
    AnomalyEvent,
    scores_to_events,
    merge_events,
    filter_events,
)

from .thresholding import (
    ThresholdConfig,
    select_threshold,
    evaluate_threshold,
)

__all__ = [
    "AnomalyEvent",
    "scores_to_events",
    "merge_events",
    "filter_events",
    "ThresholdConfig",
    "select_threshold",
    "evaluate_threshold",
]