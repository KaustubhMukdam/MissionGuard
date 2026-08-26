# tests/test_bridge.py
"""Tests for app/data_bridge.py (backend -> frontend pipeline bridge).

Uses a hermetic synthetic OPSSAT-AD-shaped workspace (segments.csv, dataset.csv,
prod artifacts) built in tmp_path, so tests are fast, deterministic, and do not
depend on the real 300K-row dataset or saved production models.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.missionguard.detection.events import AnomalyEvent
from src.missionguard.incidents.evidence import build_evidence_packet, validate_evidence_packet
from app.data_bridge import (
    MAX_TREND_POINTS,
    briefing_from_packet,
    build_dashboard_view,
    build_model_report,
    event_window_series,
    incidents_table_rows,
    load_evaluation_metrics,
    load_production_models,
    run_pipeline,
    telemetry_slice,
)

FEATURE_NAMES = [
    "duration", "len", "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
    "gaps_squared", "len_weighted", "var_div_duration", "var_div_len",
]

INT_FEATURES = {
    "duration", "len", "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks",
}

N_TRAIN = 8
N_TEST_NORMAL = 3
ROWS_PER_SEGMENT = 10


def _feature_row(rng, extreme=False):
    if extreme:
        return {
            "duration": 10, "len": 10,
            "mean": 0.9, "var": 0.05, "std": 0.22,
            "kurtosis": 8.0, "skew": 2.5,
            "n_peaks": 40, "smooth10_n_peaks": 35, "smooth20_n_peaks": 30,
            "diff_peaks": 60, "diff2_peaks": 90,
            "diff_var": 1e-4, "diff2_var": 1e-4,
            "gaps_squared": 900.0, "len_weighted": 10.0,
            "var_div_duration": 5e-6, "var_div_len": 5e-6,
        }
    return {
        "duration": 10, "len": 10,
        "mean": float(1e-6 + rng.uniform(-1e-8, 1e-8)),
        "var": float(1e-12 + rng.uniform(-1e-14, 1e-14)),
        "std": float(1e-6 + rng.uniform(-1e-9, 1e-9)),
        "kurtosis": float(-0.5 + rng.uniform(-0.01, 0.01)),
        "skew": float(0.05 + rng.uniform(-0.001, 0.001)),
        "n_peaks": 1, "smooth10_n_peaks": 1, "smooth20_n_peaks": 1,
        "diff_peaks": 2, "diff2_peaks": 3,
        "diff_var": float(1e-13 + rng.uniform(-1e-15, 1e-15)),
        "diff2_var": float(1e-13 + rng.uniform(-1e-15, 1e-15)),
        "gaps_squared": 100.0, "len_weighted": 10.0,
        "var_div_duration": float(1e-13 + rng.uniform(-1e-15, 1e-15)),
        "var_div_len": float(1e-13 + rng.uniform(-1e-15, 1e-15)),
    }


def _build_workspace(tmp_path, n_test_rows=None, inf_feature_in=None):
    """Create synthetic data/, models/, artifacts/ directories."""
    rng = np.random.RandomState(42)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    models_dir.mkdir()
    artifacts_dir.mkdir()

    n_segments = N_TRAIN + N_TEST_NORMAL + 1  # last test segment is extreme
    base_time = pd.Timestamp("2022-06-01 00:00:00", tz="UTC")

    seg_rows = []
    ds_rows = []
    sample_idx = 0
    for seg in range(1, n_segments + 1):
        is_train = seg <= N_TRAIN
        # Segment ids: 1..N_TRAIN are train, rest test; final one is the extreme one
        extreme = seg == n_segments
        label = "anomaly" if extreme else "normal"
        start = base_time + pd.Timedelta(seconds=sample_idx)
        for r in range(ROWS_PER_SEGMENT):
            seg_rows.append({
                "channel": "CH1",
                "timestamp": (start + pd.Timedelta(seconds=r)).isoformat(),
                "value": float(1e-6 + rng.uniform(-1e-7, 1e-7)) if not extreme else 0.9 - r * 0.05,
                "label": label,
                "sampling": 1,
                "anomaly": 1 if extreme else 0,
                "segment": seg,
                "train": 1 if is_train else 0,
            })
            sample_idx += 1

        feats = _feature_row(rng, extreme=extreme)
        if inf_feature_in is not None and not is_train and seg == N_TRAIN + 1:
            feats[inf_feature_in] = np.inf
        ds_rows.append({
            "segment": seg,
            "anomaly": 1 if extreme else 0,
            "train": 1 if is_train else 0,
            "channel": "CH1",
            "sampling": 1,
            **feats,
        })

    segments_df = pd.DataFrame(seg_rows)
    dataset_df = pd.DataFrame(ds_rows)

    # Enforce schema dtypes (ints without decimals)
    for col in ["sampling", "anomaly", "segment", "train"]:
        segments_df[col] = segments_df[col].astype("int64")
    for col in ["segment", "anomaly", "train", "sampling"] + sorted(INT_FEATURES):
        dataset_df[col] = dataset_df[col].astype("int64")
    for col in FEATURE_NAMES:
        if col not in INT_FEATURES:
            dataset_df[col] = dataset_df[col].astype("float64")

    if n_test_rows is not None:
        # Simulate an empty/short test split by forcing everything into train
        dataset_df.loc[dataset_df["segment"] > N_TRAIN, "train"] = 1
        segments_df.loc[segments_df["segment"] > N_TRAIN, "train"] = 1

    segments_df.to_csv(data_dir / "segments.csv", index=False)
    dataset_df.to_csv(data_dir / "dataset.csv", index=False)

    # Fit tiny production-equivalent artifacts on the train split only
    from src.missionguard.models.isolation_forest import IsolationForestDetector
    from src.missionguard.preprocessing.transforms import RobustScalerWrapper

    train_ds = dataset_df[dataset_df["train"] == 1]
    scaler = RobustScalerWrapper(FEATURE_NAMES).fit(train_ds)
    train_scaled = scaler.transform(train_ds)

    detector = IsolationForestDetector(n_estimators=20, contamination=0.05, random_state=42)
    detector.fit(train_scaled)
    train_scores = detector.score(train_scaled)
    detector.set_threshold_from_scores(train_scores, method="percentile", value=90)
    detector.save(str(models_dir / "isolation_forest_prod_v1.joblib"))
    scaler.save(str(models_dir / "robust_scaler_prod_v1.joblib"))

    config = {
        "model_name": "IsolationForestDetector",
        "version": "test-1.0.0",
        "experiment_id": "EXP-TEST",
        "feature_names": FEATURE_NAMES,
        "scaler_type": "robust",
        "score_normalization": "minmax",
        "threshold_method": "percentile",
        "threshold_value": detector.threshold,
    }
    (models_dir / "prod_config_v1.json").write_text(json.dumps(config, indent=2))

    metrics = {
        "precision": 0.8, "recall": 0.75, "f1": 0.77, "roc_auc": 0.9,
        "false_alarms_per_hour": 1.5, "mean_detection_delay_seconds": 2.0,
    }
    metrics_path = artifacts_dir / "prod_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return data_dir, models_dir, metrics_path


@pytest.fixture
def workspace(tmp_path):
    return _build_workspace(tmp_path)


def test_load_production_models_roundtrip(tmp_path, workspace):
    data_dir, models_dir, metrics_path = workspace
    model, scaler, config = load_production_models(models_dir)
    assert model.fitted
    assert model.feature_names == FEATURE_NAMES
    assert model.threshold is not None
    assert scaler.fitted
    assert config["model_name"] == "IsolationForestDetector"


def test_load_evaluation_metrics(workspace):
    _, _, metrics_path = workspace
    metrics = load_evaluation_metrics(metrics_path)
    assert metrics["f1"] == pytest.approx(0.77)
    assert load_evaluation_metrics(metrics_path.parent / "missing.json") is None
    assert load_evaluation_metrics(None) is None


class TestRunPipeline:
    def test_happy_path_structure(self, workspace):
        result = run_pipeline(*workspace)
        assert {"segments", "dataset", "scored", "events", "incidents",
                "packets", "model_info", "evaluation_metrics", "config",
                "dropped_rows"} <= set(result.keys())
        assert len(result["scored"]) == N_TEST_NORMAL + 1
        assert result["dropped_rows"] == 0
        assert len(result["incidents"]) >= 1
        assert result["model_info"]["name"] == "IsolationForestDetector"
        assert result["model_info"]["version"] == "test-1.0.0"
        assert result["evaluation_metrics"]["f1"] == pytest.approx(0.77)

    def test_extreme_segment_detected(self, workspace):
        result = run_pipeline(*workspace)
        detected_segments = set()
        for event in result["events"]:
            detected_segments.update(event.segment_ids)
        extreme_segment_id = N_TRAIN + N_TEST_NORMAL + 1
        assert extreme_segment_id in detected_segments

    def test_events_respect_threshold(self, workspace):
        result = run_pipeline(*workspace)
        threshold = result["model_info"]["threshold"]
        assert threshold is not None
        assert all(e.max_score >= threshold for e in result["events"])

    def test_incidents_have_priority_and_are_ranked(self, workspace):
        result = run_pipeline(*workspace)
        incidents = result["incidents"]
        priorities = [inc.priority_score for inc in incidents]
        assert all(p is not None for p in priorities)
        assert priorities == sorted(priorities, reverse=True)

    def test_packets_match_incidents_and_validate_clean(self, workspace):
        result = run_pipeline(*workspace)
        assert set(result["packets"].keys()) == {inc.incident_id for inc in result["incidents"]}
        for packet in result["packets"].values():
            assert packet.model_name == "IsolationForestDetector"
            assert packet.evaluation_f1 == pytest.approx(0.77)
            assert packet.priority_score is not None
            assert validate_evidence_packet(packet) == []

    def test_empty_test_split_returns_empty_results(self, tmp_path):
        data_dir, models_dir, metrics_path = _build_workspace(tmp_path, n_test_rows=0)
        result = run_pipeline(data_dir, models_dir, metrics_path)
        assert result["scored"].empty
        assert result["events"] == []
        assert result["incidents"] == []
        assert result["packets"] == {}
        assert result["dropped_rows"] == 0

    def test_missing_model_file_raises(self, workspace):
        data_dir, models_dir, metrics_path = workspace
        (models_dir / "isolation_forest_prod_v1.joblib").unlink()
        with pytest.raises(FileNotFoundError):
            run_pipeline(data_dir, models_dir, metrics_path)

    def test_nonfinite_features_dropped_not_fatal(self, tmp_path):
        data_dir, models_dir, metrics_path = _build_workspace(
            tmp_path, inf_feature_in="diff_var"
        )
        result = run_pipeline(data_dir, models_dir, metrics_path)
        assert result["dropped_rows"] == 1
        dropped_segment = N_TRAIN + 1
        assert dropped_segment not in result["scored"]["segment"].values
        # Pipeline still completes on remaining clean rows
        assert isinstance(result["events"], list)


class TestDashboardView:
    @pytest.fixture
    def view(self, workspace):
        return build_dashboard_view(run_pipeline(*workspace))

    def test_expected_keys(self, view):
        assert {"mission_name", "run_id", "model_version", "health_status",
                "active_incidents", "kpis", "trend_chart", "anomaly_log",
                "incidents"} <= set(view.keys())
        assert view["mission_name"] == "OPSSAT-AD"

    def test_incident_cards_shape(self, view):
        cards = view["incidents"]
        assert len(cards) >= 1
        scores = [c["score"] for c in cards]
        assert all(0 <= s <= 100 for s in scores)
        assert scores == sorted(scores, reverse=True)
        for card in cards:
            assert set(card.keys()) == {
                "incident_id", "start_time", "duration", "channels", "score", "priority"
            }
            assert card["priority"] in {"critical", "high", "watch", "nominal"}

    def test_anomaly_log_entries(self, view):
        for entry in view["anomaly_log"]:
            assert set(entry.keys()) == {"time", "event", "detail", "severity"}
            assert entry["severity"] in {"error", "warning", "primary", "nominal"}
        times = [e["time"] for e in view["anomaly_log"]]
        assert times == sorted(times, reverse=True)  # newest first

    def test_trend_chart_arrays_aligned_and_capped(self, view):
        chart = view["trend_chart"]
        assert len(chart["timestamps"]) == len(chart["values"])
        assert len(chart["timestamps"]) <= MAX_TREND_POINTS
        assert all(p < len(chart["values"]) for p in chart["anomaly_positions"])
        assert chart["channel"] == "CH1"

    def test_kpis_present(self, view):
        assert set(view["kpis"].keys()) == {
            "active_incidents", "anomaly_events", "model_f1", "segments_scored"
        }
        f1_kpi = view["kpis"]["model_f1"]
        assert f1_kpi["value"] == "0.770"  # from fixture metrics file

    def test_empty_pipeline_view_is_nominal(self, tmp_path):
        data_dir, models_dir, metrics_path = _build_workspace(tmp_path, n_test_rows=0)
        view = build_dashboard_view(run_pipeline(data_dir, models_dir, metrics_path))
        assert view["health_status"] == "NOMINAL"
        assert view["active_incidents"] == 0
        assert view["incidents"] == []
        assert view["anomaly_log"] == []
        assert view["trend_chart"]["timestamps"] == []
        assert view["kpis"]["active_incidents"]["value"] == "0"


class TestTelemetrySlice:
    def test_slice_alignment_and_anomaly_positions(self, workspace):
        result = run_pipeline(*workspace)
        scored = result["scored"]
        threshold = result["model_info"]["threshold"]
        sl = telemetry_slice(scored, 0, len(scored), threshold)
        assert len(sl["timestamps"]) == len(sl["values"]) == len(sl["scores"]) == len(scored)
        assert all(0 <= p < len(sl["values"]) for p in sl["anomaly_positions"])

    def test_partial_slice(self, workspace):
        result = run_pipeline(*workspace)
        scored = result["scored"]
        sl = telemetry_slice(scored, 1, 3, None)
        assert len(sl["values"]) == 2

    def test_empty_scored_frame(self, tmp_path):
        data_dir, models_dir, metrics_path = _build_workspace(tmp_path, n_test_rows=0)
        result = run_pipeline(data_dir, models_dir, metrics_path)
        sl = telemetry_slice(result["scored"], 0, 10, 0.5)
        assert sl == {"timestamps": [], "values": [], "scores": [], "anomaly_positions": []}


class TestIncidentsTableRows:
    def test_row_shape_and_ordering(self, workspace):
        result = run_pipeline(*workspace)
        rows = incidents_table_rows(result)
        assert len(rows) == len(result["incidents"])
        for row in rows:
            assert set(row.keys()) == {
                "Priority", "Incident ID", "Start", "Duration", "Channels", "Score"
            }
            assert isinstance(row["Score"], int) and 0 <= row["Score"] <= 100
            assert row["Priority"] in {"CRITICAL", "HIGH", "WATCH", "NOMINAL"}
        scores = [r["Score"] for r in rows]
        assert scores == sorted(scores, reverse=True)


class TestEventWindowSeries:
    def test_window_contains_raw_telemetry(self, workspace):
        result = run_pipeline(*workspace)
        packet = next(iter(result["packets"].values()))
        window = event_window_series(result, packet)
        assert len(window["timestamps"]) == len(window["values"])
        assert len(window["timestamps"]) > 0
        assert window["event_start"] is not None
        assert 0 <= window["event_start"] < len(window["timestamps"])

    def test_unknown_channel_returns_empty(self, workspace):
        result = run_pipeline(*workspace)
        packet = next(iter(result["packets"].values()))
        packet.affected_channels = ["DOES-NOT-EXIST"]
        window = event_window_series(result, packet)
        assert window["timestamps"] == []
        assert window["event_start"] is None


class TestBriefingFromPacket:
    def test_deterministic_and_evidence_grounded(self, workspace):
        result = run_pipeline(*workspace)
        packet = next(iter(result["packets"].values()))
        b1 = briefing_from_packet(packet)
        b2 = briefing_from_packet(packet)
        assert b1 == b2
        assert set(b1.keys()) == {"summary", "why_flagged", "suggestions"}
        assert packet.affected_channels[0] in b1["summary"]
        assert str(packet.model_name) in b1["why_flagged"]
        assert len(b1["suggestions"]) == 3


class TestModelReport:
    def test_report_from_fixture_artifacts(self, workspace):
        _, models_dir, metrics_path = workspace
        report = build_model_report(models_dir, metrics_path)
        assert report["model_config"]["model_name"] == "IsolationForestDetector"
        assert report["evaluation_metrics"]["f1"] == pytest.approx(0.77)
        # fixture has no artifacts dir next to models/, so no experiment CSV section
        assert "feature_group_experiments" not in report

    def test_missing_config_raises(self, workspace):
        _, models_dir, metrics_path = workspace
        (models_dir / "prod_config_v1.json").unlink()
        with pytest.raises(FileNotFoundError):
            build_model_report(models_dir, metrics_path)

    def test_missing_metrics_tolerated(self, workspace):
        _, models_dir, _ = workspace
        report = build_model_report(models_dir, models_dir / "missing.json")
        assert report["evaluation_metrics"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
