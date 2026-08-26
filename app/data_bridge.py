# app/data_bridge.py
"""Bridge between the MissionGuard backend pipeline and the Streamlit frontend.

Loads the production artifacts (Isolation Forest v1 + RobustScaler v1), scores the
OPSSAT-AD test split, converts scores into anomaly events, aggregates them into
priority-ranked incidents, and builds evidence packets ready for the UI and the
LLM briefing layer.

Pure Python (no Streamlit imports) so it is unit-testable in isolation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.missionguard.data.loaders import get_train_test_split, load_opssat_ad
from src.missionguard.detection.events import AnomalyEvent
from src.missionguard.incidents.aggregation import (
    Incident,
    aggregate_events_to_incidents,
)
from src.missionguard.incidents.evidence import (
    EvidencePacket,
    build_evidence_packet,
)
from src.missionguard.incidents.priority import (
    PriorityScore,
    PriorityScorer,
    get_priority_label,
)
from src.missionguard.detection.events import get_events_per_channel
from src.missionguard.models.isolation_forest import IsolationForestDetector
from src.missionguard.preprocessing.transforms import (
    RobustScalerWrapper,
    transform_features,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "opssat-ad"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "phase3b" / "prod_baseline_metrics.json"


def load_production_models(models_dir: Path):
    """Load production model, scaler, and config. Raises FileNotFoundError if missing."""
    models_dir = Path(models_dir)
    config = json.loads((models_dir / "prod_config_v1.json").read_text())
    model = IsolationForestDetector.load(str(models_dir / "isolation_forest_prod_v1.joblib"))
    scaler = RobustScalerWrapper.load(str(models_dir / "robust_scaler_prod_v1.joblib"))
    return model, scaler, config


def load_evaluation_metrics(metrics_path: Optional[Path]) -> Optional[Dict[str, float]]:
    """Load saved evaluation metrics; returns None if unavailable (never fatal)."""
    if metrics_path is None:
        return None
    path = Path(metrics_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_pipeline(
    data_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    max_gap_seconds: float = 300.0,
    min_event_samples: int = 1,
) -> Dict[str, Any]:
    """
    Run the full detection-to-incident pipeline on the OPSSAT-AD test split.

    Args:
        data_dir: Directory containing segments.csv and dataset.csv
        models_dir: Directory containing prod model/scaler/config artifacts
        metrics_path: Optional path to saved evaluation metrics JSON
        max_gap_seconds: Temporal gap for merging events into incidents
        min_event_samples: Minimum consecutive anomalous samples per event

    Returns:
        Dict with raw data, scored segments, events, ranked incidents,
        evidence packets keyed by incident_id, model info, and evaluation metrics.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    resolved_metrics = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH

    model, scaler, config = load_production_models(models_dir)
    segments, dataset = load_opssat_ad(str(data_dir))
    _, _, _, test_ds = get_train_test_split(segments, dataset)

    feature_names = model.feature_names
    model_info = {
        "name": config.get("model_name"),
        "version": config.get("version"),
        "experiment_id": config.get("experiment_id"),
        "threshold": model.threshold,
        "score_normalization": config.get("score_normalization"),
    }
    evaluation_metrics = load_evaluation_metrics(resolved_metrics)

    # Per-segment time windows + mean raw value from raw telemetry
    # (dataset.csv already carries `channel`; dataset.csv has no timestamps)
    windows = segments.groupby("segment").agg(
        timestamp=("timestamp", "min"),
        end=("timestamp", "max"),
        value=("value", "mean"),
    )
    scored = (
        test_ds.merge(windows, left_on="segment", right_index=True, how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    events: List[AnomalyEvent] = []
    incidents: List[Incident] = []
    packets: Dict[str, EvidencePacket] = {}
    dropped_rows = 0

    if not scored.empty:
        # Drop non-finite rows BEFORE scaling/scoring: RobustScaler and sklearn
        # raise on NaN/inf, and loader validation only warns about inf.
        finite_mask = np.isfinite(
            scored[feature_names].to_numpy(dtype=float)
        ).all(axis=1)
        scored = scored[finite_mask].reset_index(drop=True)
        dropped_rows = int((~finite_mask).sum())

        if not scored.empty:
            transformed = transform_features(scored, scaler, feature_names)
            scores = model.score(transformed)
            scored["anomaly_score"] = scores
            events_by_channel = get_events_per_channel(
                scored,
                scores,
                threshold=model.threshold,
                min_duration=min_event_samples,
            )
            events = [e for channel_events in events_by_channel.values() for e in channel_events]
            incidents = aggregate_events_to_incidents(events, max_gap_seconds=max_gap_seconds)

            scorer = PriorityScorer()
            scorer.rank_incidents(incidents)
            packets = {
                inc.incident_id: build_evidence_packet(
                    inc,
                    priority=PriorityScore(
                        total_score=inc.priority_score,
                        components=inc.priority_components,
                        weights=scorer.weights,
                    ),
                    model_info=model_info,
                    evaluation_metrics=evaluation_metrics,
                )
                for inc in incidents
            }

    return {
        "segments": segments,
        "dataset": dataset,
        "scored": scored,
        "events": events,
        "incidents": incidents,
        "packets": packets,
        "model_info": model_info,
        "evaluation_metrics": evaluation_metrics,
        "config": config,
        "dropped_rows": dropped_rows,
    }


MAX_TREND_POINTS = 600
MAX_LOG_ENTRIES = 6
MAX_INCIDENT_CARDS = 5
_SEVERITY_BANDS = [
    (0.75, "error"),
    (0.5, "warning"),
    (0.25, "primary"),
]


def _severity_for_score(score: float) -> str:
    for cutoff, severity in _SEVERITY_BANDS:
        if score >= cutoff:
            return severity
    return "nominal"


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def build_dashboard_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shape a run_pipeline() result into exactly the dict the Mission Overview page renders.

    Pure function over the pipeline result — no Streamlit, no I/O — so the mapping
    from real incidents/events to UI cards is unit-testable.
    """
    incidents = result["incidents"]
    events = result["events"]
    scored = result["scored"]
    model_info = result["model_info"]
    evaluation_metrics = result.get("evaluation_metrics") or {}

    top_priority = get_priority_label(incidents[0].priority_score) if incidents else "NOMINAL"

    # Trend chart: per-segment mean telemetry, capped for render performance
    if not scored.empty:
        step = max(1, len(scored) // MAX_TREND_POINTS)
        trend = scored.iloc[::step]
        threshold = model_info.get("threshold")
        anomaly_positions = (
            set(trend.index[trend["anomaly_score"] >= threshold])
            if threshold is not None
            else set()
        )
        trend_chart = {
            "timestamps": list(trend["timestamp"]),
            "values": [float(v) for v in trend["value"]],
            "anomaly_positions": sorted(anomaly_positions),
            "channel": str(trend["channel"].iloc[0]),
        }
    else:
        trend_chart = {"timestamps": [], "values": [], "anomaly_positions": [], "channel": ""}

    anomaly_log = [
        {
            "time": e.start_time.strftime("%H:%M:%S"),
            "event": f"Anomaly on {e.channel}",
            "detail": (
                f"score {e.max_score:.2f}, duration {e.duration_seconds:.0f}s, "
                f"segments {','.join(map(str, e.segment_ids[:3]))}"
                + ("…" if len(e.segment_ids) > 3 else "")
            ),
            "severity": _severity_for_score(e.max_score),
        }
        for e in sorted(events, key=lambda ev: ev.start_time, reverse=True)[:MAX_LOG_ENTRIES]
    ]

    incident_cards = [
        {
            "incident_id": inc.incident_id,
            "start_time": inc.start_time.strftime("%H:%M:%S"),
            "duration": _format_duration(inc.duration_seconds),
            "channels": inc.affected_channels,
            "score": int(round(inc.priority_score * 100)),
            "priority": get_priority_label(inc.priority_score).lower(),
        }
        for inc in incidents[:MAX_INCIDENT_CARDS]
    ]

    f1 = evaluation_metrics.get("f1")
    kpis = {
        "active_incidents": {
            "value": str(len(incidents)),
            "trend": f"top score {int(round(incidents[0].priority_score * 100))}" if incidents else "all clear",
            "variant": "error" if incidents else "nominal",
            "icon": "warning",
            "sparkline": [round(i.priority_score * 100) for i in incidents[:6]],
        },
        "anomaly_events": {
            "value": str(len(events)),
            "trend": f"{len(scored)} segments scored",
            "variant": "warning" if events else "nominal",
            "icon": "monitoring",
            "sparkline": [float(round(e.max_score, 2)) for e in events[:6]],
        },
        "model_f1": {
            "value": f"{f1:.3f}" if f1 is not None else "N/A",
            "trend": model_info.get("name") or "",
            "variant": "primary",
            "icon": "analytics",
            "sparkline": None,
        },
        "segments_scored": {
            "value": str(len(scored)),
            "trend": f"{result.get('dropped_rows', 0)} rows dropped",
            "variant": "primary",
            "icon": "dataset",
            "sparkline": None,
        },
    }

    return {
        "mission_name": "OPSSAT-AD",
        "run_id": model_info.get("experiment_id") or "PROD-V1",
        "model_version": f"{model_info.get('name', '')} {model_info.get('version', '')}".strip(),
        "health_status": top_priority,
        "active_incidents": len(incidents),
        "kpis": kpis,
        "trend_chart": trend_chart,
        "anomaly_log": anomaly_log,
        "incidents": incident_cards,
    }


def telemetry_slice(
    scored: "pd.DataFrame",
    start: int,
    end: int,
    threshold: Optional[float],
) -> Dict[str, Any]:
    """
    Slice scored segments for the Telemetry Explorer chart.

    Returns aligned lists (timestamps, values, scores) plus anomaly positions
    (indices into the slice where score >= threshold).
    """
    empty = {"timestamps": [], "values": [], "scores": [], "anomaly_positions": []}
    if scored is None or scored.empty:
        return empty
    sl = scored.iloc[start:end]
    if sl.empty:
        return empty
    positions = set()
    if threshold is not None:
        positions = set(sl.index[sl["anomaly_score"] >= threshold] - sl.index[0])
    return {
        "timestamps": list(sl["timestamp"]),
        "values": [float(v) for v in sl["value"]],
        "scores": [float(s) for s in sl["anomaly_score"]],
        "anomaly_positions": sorted(int(p) for p in positions),
    }


def incidents_table_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map ranked incidents to rows for the Incident Center table (st.dataframe)."""
    return [
        {
            "Priority": get_priority_label(inc.priority_score),
            "Incident ID": inc.incident_id,
            "Start": inc.start_time.strftime("%H:%M:%S"),
            "Duration": _format_duration(inc.duration_seconds),
            "Channels": ", ".join(inc.affected_channels),
            "Score": int(round(inc.priority_score * 100)),
        }
        for inc in result["incidents"]
    ]


def event_window_series(result: Dict[str, Any], packet: EvidencePacket) -> Dict[str, Any]:
    """
    Extract raw telemetry within an incident's time window for the Autopsy chart.

    Returns {timestamps, values, event_start} — event_start marks the first
    anomalous sample position inside the window.
    """
    segments = result["segments"]
    channel = packet.affected_channels[0] if packet.affected_channels else None
    start = pd.Timestamp(packet.start_time)
    end = pd.Timestamp(packet.end_time)

    window = segments[
        (segments["channel"] == channel)
        & (segments["timestamp"] >= start)
        & (segments["timestamp"] <= end)
    ].sort_values("timestamp")

    if window.empty:
        return {"timestamps": [], "values": [], "event_start": None}

    first_event = min(
        (pd.Timestamp(e["start_time"]) for e in packet.anomaly_events if e.get("start_time")),
        default=None,
    )
    event_start = None
    if first_event is not None:
        matches = window.index[window["timestamp"] >= first_event]
        if len(matches):
            event_start = int(window.index.get_loc(matches[0]))

    return {
        "timestamps": list(window["timestamp"]),
        "values": [float(v) for v in window["value"]],
        "event_start": event_start,
    }


def briefing_from_packet(packet: EvidencePacket) -> Dict[str, Any]:
    """
    Deterministic operator briefing built ONLY from evidence-packet fields.

    This is the no-LLM fallback required by the architecture doc: every sentence
    traces to structured evidence. The Granite layer will replace/augment this.
    """
    channels = ", ".join(packet.affected_channels) or "unknown channel"
    threshold_txt = (
        f"{packet.threshold_used:.3f}" if packet.threshold_used else "the deployed threshold"
    )
    summary = (
        f"{packet.event_count} anomaly event(s) detected on {channels} between "
        f"{packet.start_time[11:19]}Z and {packet.end_time[11:19]}Z "
        f"(duration {packet.duration_seconds:.0f}s)."
    )
    why_flagged = (
        f"Model '{packet.model_name}' scored up to {packet.max_anomaly_score:.2f}, "
        f"exceeding deployed threshold {threshold_txt}."
    )
    top_segment = ""
    if packet.anomaly_events and packet.anomaly_events[0].get("segment_ids"):
        ids = packet.anomaly_events[0]["segment_ids"]
        top_segment = f" (segments {','.join(str(i) for i in ids[:3])})"
    suggestions = [
        f"Inspect raw telemetry for {channels}{top_segment} around the event window.",
        f"Compare affected channel statistics against nominal baseline "
        f"(max score {packet.max_anomaly_score:.2f}, mean {packet.mean_anomaly_score:.2f}).",
        f"Check for recurring events on the same channels; priority score was "
        f"{packet.priority_score:.2f} ({packet.priority_label}).",
    ]
    return {
        "summary": summary,
        "why_flagged": why_flagged,
        "suggestions": suggestions,
    }


def build_model_report(
    models_dir: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assemble a JSON-serializable model report from production artifacts."""
    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    resolved_metrics = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH

    config = json.loads((models_dir / "prod_config_v1.json").read_text())
    report = {
        "generated_at": datetime.now().isoformat(),
        "model_config": config,
        "evaluation_metrics": load_evaluation_metrics(resolved_metrics),
    }
    feature_groups = models_dir.parent / "artifacts" / "phase3b" / "experiment_feature_groups.csv"
    if feature_groups.exists():
        import csv as _csv
        with open(feature_groups) as f:
            report["feature_group_experiments"] = list(_csv.DictReader(f))
    return report
