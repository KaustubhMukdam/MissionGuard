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