# Learnings — MissionGuard

> Update after every meaningful development session. Write in your own words.

---

## 2026-08-12 — Ideation and dataset selection

### What I learned

**Dataset selection is part of ML system design.** The dataset determines which claims the product can legitimately make. A single-channel anomaly benchmark is not enough to justify multivariate subsystem reasoning.

**Evidence-grounded LLM design:** the LLM should sit after deterministic/ML analysis and summarize a structured evidence packet instead of being asked to diagnose raw telemetry.

### Code snippet that clicked

```python
incident = {
    "incident_id": "INC-001",
    "start_time": "...",
    "end_time": "...",
    "channels": ["..."],
    "anomaly_scores": [0.91],
    "priority_score": 0.78,
}
```

The important idea is that the LLM receives structured evidence, not an unstructured telemetry dump.

### What confused me today

How the exact ESA-ADB subset is represented locally and how its labels map to anomaly events.

### How I solved it

The project decision is to perform a dedicated dataset-validation step before model implementation. The official ESA-ADB repository documents the dataset structure, evaluation pipeline, and preprocessing workflow.

### What I'd do differently

Do not choose a neural architecture before inspecting the dataset and reproducing a simple baseline.

---

## 2026-08-15 — Phase 0: Dataset Validation (OPSSAT-AD)

### What I learned

**OPSSAT-AD is a viable fallback but has limitations for the full MissionGuard vision:**
- 9 channels available, but they are NOT time-aligned — each segment contains data from a single channel
- No channel metadata (subsystem, unit, description) — cannot do cross-channel/subsystem aggregation honestly
- Train/test split is by segment, not by time — leakage risk if not careful
- Anomaly labels are segment-level with 4 types ('anomaly', 'a2', 'a3', 'a4') — supports multi-class but segment granularity
- 33% anomaly rate at point level is HIGH — likely because segments are pre-segmented around events

**ESA-ADB Mission 1 3-month subset (262K rows, 87 channels) is the real target** for multivariate incident aggregation. The TimeEval DatasetManager should provide access without downloading 3.7 GB.

**Key architectural implications:**
- Statistical baseline should work on segment features (dataset.csv) first
- Isolation Forest on 18 segment features is a natural fit
- For raw time series (segments.csv), need to handle non-monotonic timestamps (group by segment first)
- Cross-channel aggregation requires ESA-ADB or strong assumptions
- Priority scoring can use segment features (duration, anomaly score, feature magnitudes)

### Code snippet that clicked

```python
# Segments are not time-ordered globally — must sort by segment then timestamp
segments_sorted = segments.sort_values(['segment', 'timestamp'])

# Train/test split by segment ID (provided)
train_segments = segments[segments['train'] == 1]
test_segments = segments[segments['train'] == 0]
```

### What confused me today

Why timestamps are not globally monotonic in segments.csv — realized it's because segments from different channels/time periods are concatenated, not interleaved chronologically.

### How I solved it

Group by segment first, then analyze within-segment time series. For global temporal analysis, need to sort by timestamp across all segments.

### What I'd do differently

Start with segment-level features (dataset.csv) for baseline experiments — cleaner, no timestamp issues, matches benchmark protocol. Use raw time series only for visualization and temporal incident aggregation later.