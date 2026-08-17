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

The validation functions failed when test data had missing columns — fixed by checking only present columns for null validation, while still reporting missing required columns as schema errors.

### How I solved it

Modified `validate_segments_df` and `validate_dataset_df` to check nulls only on columns that exist in the DataFrame, while still flagging missing required columns as schema errors.

### What I'd do differently

Add a configuration-driven approach for feature selection earlier. The 18 segment features in OPSSAT-AD are pre-computed; for ESA-ADB we'll need to compute similar features from raw multivariate data.

---

## 2026-08-17 — Phase 2: Baseline Anomaly Detection

### What I learned

**Statistical baselines first:**
- Implemented `StatisticalBaseline` (global MAD/Z-score), `RollingMADBaseline`, `RollingZScoreBaseline`
- Global statistical baselines on segment features are fast but don't capture temporal dynamics
- Rolling baselines need careful window/min_periods tuning for telemetry sampling rates
- MAD is more robust than Z-score for telemetry with outliers

**Isolation Forest as lightweight ML baseline:**
- `IsolationForestDetector` with 18 segment features trains in <1s
- Score normalization (minmax, percentile) maps raw scores to [0,1] for thresholding
- `contamination="auto"` works well; explicit values (0.01-0.1) give more control
- Feature name tracking prevents column mismatch errors

**Base detector pattern:**
- Abstract `BaseAnomalyDetector` enforces fit/score/predict interface
- Threshold tuning on validation set (F1-optimal or percentile-based)
- Persistence via joblib (model + threshold + feature names)
- Prevents leakage by design (fit on train, threshold on val, evaluate on test)

**Score-to-event conversion:**
- `scores_to_events` converts continuous scores → discrete events with timestamps, duration, max/mean scores
- `min_duration` filters single-sample spikes
- `merge_events` combines nearby events within configurable gap (default 5 min)
- `filter_events` by duration, max_score, mean_score
- Events carry segment IDs for traceability

**Threshold selection:**
- Percentile-based (simple, unsupervised)
- F1-optimal on validation (supervised, needs labels)
- Precision@Recall / Recall@Precision for operational targets
- Sweep evaluation for threshold visualization

**Evaluation metrics:**
- Point-level: Precision, Recall, F1, Accuracy, Specificity, ROC-AUC, PR-AUC
- Operational: False alarms/hour, False alarm rate, MTBFA, Detection delay
- Bootstrap confidence intervals for metric uncertainty
- Threshold-independent metrics (ROC-AUC, PR-AUC) reported alongside threshold-dependent

**OPSSAT-AD baseline results (Isolation Forest):**
- F1: ~0.43, Precision: ~0.31, Recall: ~0.73
- High recall but low precision — expected for segment-features without temporal context
- False alarms: ~1300/hour (high due to many segments)
- Detection delay: ~0.1s (segments already grouped)
- Segment-features alone insufficient for clean separation — needs temporal context

### Code snippet that clicked

```python
# Complete baseline pipeline
from missionguard.data import load_opssat_ad, get_train_test_split
from missionguard.preprocessing import fit_scaler, transform_features, get_feature_names, prepare_features_target
from missionguard.models import IsolationForestDetector
from missionguard.evaluation import compute_all_metrics

segments, dataset = load_opssat_ad("data/raw/opssat-ad")
train_seg, test_seg, train_ds, test_ds = get_train_test_split(segments, dataset)

feature_names = get_feature_names(train_ds)
scaler = fit_scaler(train_ds, feature_names, scaler_type="robust")
train_scaled = transform_features(train_ds, scaler, feature_names)
test_scaled = transform_features(test_ds, scaler, feature_names)

X_train, y_train = prepare_features_target(train_scaled, feature_names)
X_test, y_test = prepare_features_target(test_scaled, feature_names)

detector = IsolationForestDetector(n_estimators=100, score_normalization="minmax", random_state=42)
detector.fit(X_train)
detector.tune_threshold(X_train, y_train, metric="f1")
test_scores = detector.score(X_test)

metrics = compute_all_metrics(y_test.values, test_scores, threshold=detector.threshold)
print(f"F1: {metrics.f1:.3f}, Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}")
```

### What confused me today

Statistical baseline (MAD) threshold was absurdly high (2e10) — realized MAD scores on segment features have very different scale than Isolation Forest scores. Need score normalization or percentile thresholding for statistical baselines too.

### How I solved it

Use `set_threshold_from_scores(scores, method="percentile", value=95)` for statistical baselines instead of trying to tune on labels. The score scales are not comparable across model types.

### What I'd do differently

Add score normalization to statistical baselines too, or enforce a common score interface. The current `StatisticalBaseline` returns raw MAD/Z-score values which aren't directly comparable to Isolation Forest's [0,1] normalized scores.

---

## 2026-08-17 — Phase 3a: Rolling Baselines on Raw Telemetry (OPSSAT-AD)

### What I learned

**Rolling baselines on raw telemetry add temporal context but still limited:**
- Computed rolling features (mean, std, min, max, skew, kurt) on raw telemetry per segment (windows: 10s, 30s, 60s)
- 18 rolling features per segment vs 18 static segment features — same dimensionality but temporal
- Rolling MAD/Z-Score baselines trained on 220K train rows, evaluated on 76K test rows

**Results — Rolling baselines still underperform:**
| Model | Window | F1 | Precision | Recall | Threshold |
|-------|--------|-----|-----------|--------|-----------|
| Rolling MAD | 10s | 0.077 | 0.43 | 0.04 | 18.2 |
| Rolling MAD | 30s | 0.048 | 0.29 | 0.03 | 58.7 |
| Rolling MAD | 60s | 0.096 | 0.29 | 0.06 | 56.0 |
| Rolling Z-Score | 10s | 0.071 | 0.44 | 0.04 | 2.7 |
| Rolling Z-Score | 30s | 0.068 | 0.36 | 0.04 | 3.9 |
| Rolling Z-Score | 60s | 0.098 | 0.35 | 0.06 | 4.2 |

**Key finding:** Rolling features don't significantly improve over static segment features (Isolation Forest F1=0.43 vs Rolling MAD F1=0.096). The rolling features have extreme values (skew/kurt up to 43,000) because segments have near-constant values with sudden anomaly spikes.

**Event detection works but limited:**
- Detected 9 events across 7 test segments (threshold at 95th percentile)
- Merged 9 events → 6 incidents using 5-minute temporal gap
- Max incident duration: 444s (7.4 min)
- All incidents on single channel (CADC0872) — no cross-channel incidents possible with OPSSAT-AD

**Root cause of poor performance:**
- OPSSAT-AD segments are pre-extracted around events — "cheating" for supervised learning
- Rolling features computed per-segment, not continuous across time
- 9 channels but each segment = single channel (no multivariate context)
- Rolling skew/kurt have extreme values (up to 43,000) from constant segments with brief anomalies

**Architectural implications:**
- Rolling baselines on OPSSAT-AD raw telemetry don't beat static segment features
- Need ESA-ADB for true multivariate temporal modeling
- For OPSSAT-AD, segment-level features + Isolation Forest is the practical baseline
- Incident engine works on segment-level events but limited to single-channel

### Code snippet that clicked

```python
# Rolling baseline pipeline on raw telemetry
from missionguard.data import load_opssat_ad, get_train_test_split
from missionguard.preprocessing.time_series import sort_by_segment_time, compute_rolling_features
from missionguard.models import RollingMADBaseline
from missionguard.evaluation import compute_all_metrics

segments, dataset = load_opssat_ad("data/raw/opssat-ad")
segments_sorted = sort_by_segment_time(segments)

# Compute rolling features per segment
rolling_features_list = []
for seg_id in segments_sorted["segment"].unique():
    seg_data = extract_segment_windows(segments_sorted, seg_id).reset_index(drop=True)
    rolling_df = compute_rolling_features(seg_data["value"], windows=[10,30,60], features=["mean","std","min","max","skew","kurt"])
    rolling_df["segment"] = seg_id
    # ... add metadata
    rolling_features_list.append(rolling_df)

rolling_features = pd.concat(rolling_features_list).dropna(subset=feature_cols)

# Train rolling baseline
mad = RollingMADBaseline(window=60, min_periods=5, aggregation="max")
mad.fit(X_train[window_features])
mad.set_threshold_from_scores(val_scores, method="percentile", value=95)

# Event detection
events = scores_to_events(scores, timestamps, channel, threshold=mad.threshold, min_duration=5)
incidents = merge_events(events, max_gap_seconds=300)
```

### What confused me today

Rolling MAD baseline had NaN thresholds initially — realized the validation scores contained NaN from rolling window warmup, which made `np.percentile` return NaN. Fixed by filtering NaN before percentile calculation.

### How I solved it

Added NaN filtering before threshold selection: `val_scores_valid = val_scores[~np.isnan(val_scores)]` before calling `set_threshold_from_scores`.

### What I'd do differently

For OPSSAT-AD, skip rolling baselines on raw telemetry — the segment-level Isolation Forest is better and faster. Rolling baselines only make sense with continuous multivariate telemetry (ESA-ADB).

---

## 2026-08-17 — Phase 3b: Isolation Forest Production Baseline + Extended Experiments

### What I learned

**Production baseline formalized:**
- Saved `IsolationForestDetector` with config, scaler, threshold to `models/`
- F1-optimal threshold on validation (0.108) generalizes well to test
- Production metrics: F1=0.4365, Precision=0.299, Recall=0.805, ROC-AUC=0.636, PR-AUC=0.376

**Extended experiments — Key findings:**

| Experiment | Best Config | F1 | Precision | Recall |
|------------|-------------|-----|-----------|--------|
| **Contamination sweep** | All values (0.01-0.2, auto) | **0.437** | 0.30 | 0.81 |
| **N_estimators** | 300 trees | **0.440** | 0.30 | 0.81 |
| **Feature groups** | **Peak-based (3 feat)** | **0.656** | **0.78** | **0.57** |
| **Normalization** | All (minmax/percentile/none) | **0.437** | 0.30 | 0.81 |

**🚀 MAJOR FINDING: Peak-based features (3 features) achieve F1=0.656 — 50% improvement over all 18 features!**

Peak features: `n_peaks`, `smooth10_n_peaks`, `smooth20_n_peaks`

**Other findings:**
- Contamination parameter has no effect when using F1-optimal threshold (threshold compensates)
- N_estimators: stable across 50-500, slight peak at 300
- Score normalization method doesn't matter with F1-optimal threshold
- Bootstrap 95% CI for F1: [0.390, 0.480] — stable

**Production artifacts saved:**
- Model: `models/isolation_forest_prod_v1.joblib`
- Scaler: `models/robust_scaler_prod_v1.joblib`  
- Config: `models/prod_config_v1.json`
- All experiment data: `artifacts/phase3b/`

**Production baseline pipeline now ready for Phase 4 (Incident Engine):**

```python
# Load production model
from missionguard.models import IsolationForestDetector
from missionguard.preprocessing import RobustScalerWrapper

model = IsolationForestDetector.load("models/isolation_forest_prod_v1.joblib")
scaler = RobustScalerWrapper.load("models/robust_scaler_prod_v1.joblib")

# Score new segments
X_scaled = transform_features(new_segments, scaler, feature_names)
scores = model.score(X_scaled)
events = scores_to_events(scores, timestamps, channel, threshold=model.threshold)
incidents = merge_events(events, max_gap_seconds=300)
```

### What confused me today

Peak-based features (3 features) dramatically outperforming all 18 features was unexpected — usually more features help. Realized peak counts capture the "spikiness" of anomalies directly, while statistical moments (mean, var, skew, kurt) get diluted by the long constant portions of each segment.

### How I solved it

Feature group ablation study revealed peak features as the discriminative core. The other features (statistical moments, duration, differencing) add noise without signal.

### What I'd do differently

For OPSSAT-AD, use **peak-based features only** (3 features) with Isolation Forest as the production baseline — simpler, faster, 50% better F1. This also makes the model more interpretable for operators.