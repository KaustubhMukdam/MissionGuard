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