# Experiment Log — MissionGuard

> Every model run is an experiment. Never overwrite results without recording what changed.

---

## Experiment 000 — Project setup

**Date:** 2026-08-12

**Hypothesis:** A reproducible benchmark workflow should be established before choosing the final anomaly model.

**Change made:**
- Selected ESA-ADB as the primary dataset direction.
- Chosen a lightweight development subset as the initial target.
- Defined a baseline-first model strategy.

**Results:**

| Metric | Result |
|--------|--------|
| Model | Not trained |
| Dataset | Not yet locally validated |
| F1 | N/A |
| Precision | N/A |
| Recall | N/A |
| False alarms | N/A |
| Detection delay | N/A |

**What happened:** The project is deliberately stopped before model implementation until the dataset schema, labels, timestamps, and subset feasibility are verified.

**Why (your understanding):** A model result is meaningless if the data split, label semantics, or anomaly-event definition is unclear.

**Next experiment:** EXP-001 — Dataset validation and visualization.

---

## Experiment 001 — OPSSAT-AD Dataset Validation

**Date:** 2026-08-15

**Hypothesis:** OPSSAT-AD fallback dataset is sufficient for Phase 0-3 (baseline detection, model experiments) and can be loaded/validated locally.

**Data:** OPSSAT-AD (Zenodo 12588359) — `segments.csv` (303K rows, 9 channels) + `dataset.csv` (2123 segments, 18 features)

**Change made:**
```python
# Downloaded from Zenodo
# segments.csv: raw telemetry with point-level anomaly labels
# dataset.csv: segment-level statistical features with segment-level anomaly labels
# Train/test split provided via 'train' column
```

**Results:**

| Metric | Result |
|--------|--------|
| Raw telemetry rows | 303,493 |
| Unique channels | 9 (CADC0872, CADC0892, CADC0874, CADC0884, CADC0873, CADC0886, CADC0888, CADC0894, CADC0890) |
| Timestamp range | 2022-01-04 to 2022-06-02 (~149 days) |
| Sampling rates | 1 Hz, 5 Hz |
| Anomaly rows (point) | 100,264 (33.0%) |
| Anomaly segments | 434 / 2,123 (20.4%) |
| Train segments | 1,594 |
| Test segments | 529 |
| Segment features | 18 statistical features |
| Anomaly types | 4 ('anomaly', 'a2', 'a3', 'a4') |

**What happened:** Successfully downloaded and validated OPSSAT-AD. Generated 5 exploratory plots saved to `artifacts/`. Key findings:
- Multiple channels but NOT time-aligned (each segment = single channel)
- No channel metadata (subsystem, units, descriptions)
- Timestamps not globally monotonic (segments concatenated)
- Train/test split by segment, not by time
- High anomaly rate (33%) due to segment-centric sampling

**Why (your understanding):** OPSSAT-AD is designed as a segment classification benchmark, not a multivariate time-series anomaly detection benchmark. Segments are pre-extracted around events of interest, explaining the high anomaly rate.

**Limitations:**
- Cross-channel aggregation NOT supported (no time alignment, no metadata)
- Temporal incident aggregation possible but segments already group events
- Priority scoring limited to segment-level features
- Not a substitute for ESA-ADB Mission 1 (87 channels, multivariate)

**Next experiment:** EXP-002 — Statistical baseline on OPSSAT-AD segment features (dataset.csv)

---

## Experiment 002 — Statistical Baseline (MAD) on OPSSAT-AD Segment Features

**Date:** 2026-08-17

**Hypothesis:** A simple statistical baseline (Median Absolute Deviation) on the 18 pre-computed segment features will provide a meaningful lower bound for anomaly detection performance.

**Data:** OPSSAT-AD segment features (dataset.csv), 1594 train / 529 test segments, 18 features, segment-based split

**Change made:**
```python
# StatisticalBaseline(method="mad", aggregation="max")
# Fit on train features only (no target needed for unsupervised)
# Threshold: 95th percentile of train scores
```

**Results:**

| Metric | Result |
|--------|--------|
| Model | StatisticalBaseline (MAD, max aggregation) |
| Threshold method | 95th percentile (unsupervised) |
| Test F1 | 0.44 |
| Test Precision | 0.73 |
| Test Recall | 0.32 |
| Test ROC-AUC | 0.89 |
| Test PR-AUC | 0.62 |

**What happened:** Statistical MAD baseline achieves moderate F1 with high precision but low recall. The high precision suggests MAD scores are well-calibrated for the segment features, but low recall indicates many anomalies have MAD scores below the 95th percentile threshold.

**Why (your understanding):** MAD computes deviation from rolling median in feature space. The 18 segment features (duration, statistical moments, peak counts, differencing stats) capture anomaly characteristics, but the global MAD doesn't adapt to feature-specific scales. High precision means when MAD flags something, it's usually correct; low recall means many anomalies are missed.

**Limitations:**
- Global MAD treats all features equally (no feature weighting)
- Score scale is raw MAD units — not comparable to other models
- 95th percentile threshold is arbitrary; F1-optimal would need labels
- Rolling MAD not tested yet (needs raw time series per segment)

**Next experiment:** EXP-003 — Isolation Forest on OPSSAT-AD segment features

---

## Experiment 003 — Isolation Forest on OPSSAT-AD Segment Features

**Date:** 2026-08-17

**Hypothesis:** Isolation Forest on the 18 segment features will outperform the statistical baseline by learning multivariate feature interactions.

**Data:** OPSSAT-AD segment features (dataset.csv), 1594 train / 529 test segments, 18 features, segment-based split

**Change made:**
```python
# IsolationForestDetector(
#     n_estimators=100,
#     contamination="auto",
#     score_normalization="minmax",
#     random_state=42
# )
# Threshold: F1-optimal on validation (20% of train)
```

**Results:**

| Metric | Statistical Baseline (MAD) | Isolation Forest | Change |
|--------|---------------------------|------------------|--------|
| F1 | 0.44 | 0.43 | -0.01 |
| Precision | 0.73 | 0.31 | -0.42 |
| Recall | 0.32 | 0.73 | +0.41 |
| ROC-AUC | 0.89 | 0.89 | 0.00 |
| PR-AUC | 0.62 | 0.63 | +0.01 |
| False alarms/hr | 1,300 | 1,300 | 0 |

**What happened:** Isolation Forest achieves similar F1 but with very different precision/recall trade-off. It has much higher recall (catches more anomalies) but much lower precision (many false alarms). ROC-AUC identical, meaning ranking quality is the same. The F1-optimal threshold favors recall because anomalies are rare (10% in test).

**Why (your understanding):** Isolation Forest learns multivariate boundaries in the 18D feature space. The `contamination="auto"` and minmax normalization push scores to [0,1], but the F1-optimal threshold on validation ends up very low (~0.15), triggering on many borderline segments. This increases recall but floods with false positives. The segment features alone don't provide clean separation — they're statistical summaries, not temporal patterns.

**Limitations:**
- Segment features are pre-computed statistics, not raw temporal patterns
- F1-optimal threshold depends heavily on validation set anomaly rate
- High false alarm rate (~1300/hr) operationally unacceptable
- No temporal context — each segment scored independently

**Next experiment:** EXP-004 — Rolling MAD baseline on raw telemetry per segment

---

## Experiment 004 — Rolling MAD/Z-Score Baselines on Raw Telemetry

**Date:** 2026-08-17

**Hypothesis:** Rolling statistical baselines on raw telemetry per segment will capture temporal dynamics better than static segment features.

**Data:** OPSSAT-AD raw telemetry (segments.csv), 303K rows, 2123 segments, 9 channels, segment-based split (220K train / 77K test rows)

**Change made:**
```python
# Compute rolling features per segment (windows: 10s, 30s, 60s; features: mean, std, min, max, skew, kurt)
# RollingMADBaseline / RollingZScoreBaseline on 18 rolling features
# Threshold: 95th percentile on validation
```

**Results:**

| Model | Window | F1 | Precision | Recall | Threshold |
|-------|--------|-----|-----------|--------|-----------|
| Rolling MAD | 10s | 0.077 | 0.43 | 0.04 | 18.2 |
| Rolling MAD | 30s | 0.048 | 0.29 | 0.03 | 58.7 |
| Rolling MAD | 60s | 0.096 | 0.29 | 0.06 | 56.0 |
| Rolling Z-Score | 10s | 0.071 | 0.44 | 0.04 | 2.7 |
| Rolling Z-Score | 30s | 0.068 | 0.36 | 0.04 | 3.9 |
| Rolling Z-Score | 60s | 0.098 | 0.35 | 0.06 | 4.2 |

**What happened:** Rolling baselines on raw telemetry perform WORSE than static segment features (Isolation Forest F1=0.43 vs Rolling MAD F1=0.096). Rolling features have extreme skew/kurt (up to 43,000) from near-constant segments with brief spikes.

**Why (your understanding):** OPSSAT-AD segments are pre-extracted around events. Rolling features computed per-segment, not continuous across time. No multivariate context (1 segment = 1 channel).

**Limitations:** Rolling baselines don't beat static features. Event detection works (9 events → 6 incidents) but single-channel only.

**Next experiment:** EXP-005 — Isolation Forest Production Baseline + Extended Experiments

---

## Experiment 005 — Isolation Forest Production Baseline + Extended Experiments

**Date:** 2026-08-17

**Hypothesis:** Formalize Isolation Forest as production baseline and run extended experiments to optimize performance.

**Data:** OPSSAT-AD segment features (dataset.csv), 1594 train / 529 test segments, 18 features, segment-based split

**Change made:**
```python
# Production config: IsolationForestDetector(n_estimators=200, contamination=0.1, score_normalization="minmax", random_state=42)
# RobustScaler on train, F1-optimal threshold on validation (20% of train)
# Extended experiments: contamination sweep, n_estimators sweep, feature groups, normalization, bootstrap
```

**Results — Production Baseline:**

| Metric | Value |
|--------|-------|
| F1 | 0.4365 |
| Precision | 0.299 |
| Recall | 0.805 |
| ROC-AUC | 0.636 |
| PR-AUC | 0.376 |
| Threshold (F1-optimal) | 0.108 |
| False alarms/hr | 1,450 |
| Detection delay | 0.1s |

**Extended Experiments — Key Findings:**

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
- N_estimators: stable across 50-500, peak at 300
- Score normalization method doesn't matter with F1-optimal threshold
- Bootstrap 95% CI for F1: [0.390, 0.480] — stable

**Feature Group Ablation:**

| Group | Features | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| Peak-based | n_peaks, smooth10_n_peaks, smooth20_n_peaks | **0.656** | **0.78** | **0.57** |
| Statistical | mean, var, std, kurtosis, skew | 0.337 | 0.24 | 0.57 |
| Duration | duration, len, len_weighted, gaps_squared, var_div_duration, var_div_len | 0.361 | 0.25 | 0.66 |
| Diff-based | diff_peaks, diff2_peaks, diff_var, diff2_var | 0.306 | 0.20 | 0.66 |
| All 18 | (all) | 0.437 | 0.30 | 0.81 |

**Production Artifacts Saved:**
- Model: `models/isolation_forest_prod_v1.joblib`
- Scaler: `models/robust_scaler_prod_v1.joblib`  
- Config: `models/prod_config_v1.json`
- All experiment data: `artifacts/phase3b/`

**Production Baseline Pipeline:**

```python
from missionguard.models import IsolationForestDetector
from missionguard.preprocessing import RobustScalerWrapper

model = IsolationForestDetector.load("models/isolation_forest_prod_v1.joblib")
scaler = RobustScalerWrapper.load("models/robust_scaler_prod_v1.joblib")

X_scaled = transform_features(new_segments, scaler, feature_names)
scores = model.score(X_scaled)
events = scores_to_events(scores, timestamps, channel, threshold=model.threshold)
incidents = merge_events(events, max_gap_seconds=300)
```

**Why (your understanding):** Peak features capture the "spikiness" of anomalies directly. Statistical moments get diluted by long constant portions of segments. F1-optimal threshold compensates for contamination/normalization.

**Limitations:** Still high false alarms (~1450/hr) for operational use. Peak-feature model (F1=0.656) is the new production candidate.

**Next experiment:** EXP-006 — Peak-feature Isolation Forest as final production baseline

---

## Experiment template

## Experiment [number] — [date]

**Hypothesis:** [what you expect and why]

**Data:** [mission/subset/split]

**Change made:**
```python
# Relevant code/configuration
```

**Results:**

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Precision | | | |
| Recall | | | |
| F1 | | | |
| PR-AUC | | | |
| False-alarm metric | | | |
| Detection delay | | | |
| Runtime | | | |

**What happened:** [plain-English result]

**Why (your understanding):** [mechanistic explanation]

**Limitations:** [what the result does not prove]

**Next experiment:** [next hypothesis]

---