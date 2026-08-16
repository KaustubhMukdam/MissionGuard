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

---

## 2026-08-17 — Phase 1: Data Pipeline

### What I learned

**Reusable data pipeline is the foundation:**
- Created `src/missionguard/data/` for loading and validation
- Created `src/missionguard/preprocessing/` for transforms and time series utilities
- Schema validation catches data issues early (missing columns, wrong dtypes, invalid values, missing data)
- Train/test split must respect temporal ordering for time series — added `get_temporal_train_test_split` alongside provided segment-based split
- OPSSAT-AD timestamps are NOT globally monotonic — must sort by segment then timestamp before any temporal analysis

**Scaling strategy:**
- RobustScaler preferred over StandardScaler for telemetry (outliers common)
- Scaler must be fitted ONLY on training data, then applied to test (no leakage)
- Implemented `StandardScalerWrapper` and `RobustScalerWrapper` with persistence (save/load via joblib)
- Feature names tracked explicitly to avoid column ordering issues

**Time series handling:**
- `sort_by_segment_time` essential for OPSSAT-AD (segments concatenated, not interleaved)
- `extract_segment_windows` for per-segment analysis
- `compute_rolling_features` and `compute_differencing_features` for feature engineering on raw series
- `detect_gaps` identifies sampling irregularities
- `align_channels_temporally` only works when channels overlap in time (not for OPSSAT-AD)

**Testing:**
- 31 unit tests covering schema validation, loaders, temporal splits, scalers, time series utils
- Tests use synthetic data for unit tests, real OPSSAT-AD for integration tests
- All tests pass, CI-ready

### Code snippet that clicked

```python
# Fit scaler on TRAIN only, transform both train and test
scaler = fit_scaler(train_df, feature_names, scaler_type="robust")
train_scaled = transform_features(train_df, scaler, feature_names)
test_scaled = transform_features(test_df, scaler, feature_names)  # Same scaler!

# Temporal split for time series (no future leakage)
train_df, test_df = get_temporal_train_test_split(segments_df, test_ratio=0.25)
```

### What confused me today

The validation functions failed when test data had missing columns — fixed by checking only present columns for null validation, while still reporting missing required columns as errors.

### How I solved it

Modified `validate_segments_df` and `validate_dataset_df` to check nulls only on columns that exist in the DataFrame, while still flagging missing required columns as schema errors.

### What I'd do differently

Add a configuration-driven approach for feature selection earlier. The 18 segment features in OPSSAT-AD are pre-computed; for ESA-ADB we'll need to compute similar features from raw multivariate data.