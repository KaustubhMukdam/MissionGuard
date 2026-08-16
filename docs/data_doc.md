# Data Documentation — ESA Anomaly Dataset / ESA-ADB

## Source

- **Origin:** European Space Agency satellite telemetry benchmark
- **Primary dataset:** ESA Anomaly Dataset (ESA-AD)
- **Benchmark:** ESA Anomaly Detection Benchmark (ESA-ADB)
- **Official ESA dataset:** https://github.com/esa/anomaly-dataset
- **Benchmark code:** https://github.com/kplabs-pl/ESA-ADB
- **Zenodo dataset:** https://doi.org/10.5281/zenodo.12528696
- **Kaggle challenge:** https://www.kaggle.com/competitions/esa-adb-challenge
- **License:** Verify the exact dataset license in the downloaded release before redistribution; the ESA/KP Labs dataset listings identify ESA-AD as CC-BY.
- **Downloaded:** TBD (full Mission 1 is ~3.7 GB on Zenodo)

## Structure

The official ESA-ADB documentation describes ESA-AD as real-life satellite telemetry from three ESA missions, with two missions selected for the benchmark. The raw dataset contains per-channel telemetry and separate anomaly labels.

For the first MissionGuard milestone:
- **Mission:** Mission 1
- **Subset:** Lightweight development subset; validate exact channel selection/schema locally before freezing code
- **Target:** Anomalous telemetry segments/events
- **Rows/points:** TBD after download
- **Channels:** TBD after validation

## Preprocessing to investigate

1. Load the official mission data without modifying source files.
2. Validate timestamp ordering and uniqueness.
3. Identify missing values, gaps, constant segments, and invalid values.
4. Align channels only where the benchmark's data structure supports valid alignment.
5. Read anomaly labels and verify interval semantics.
6. Create a reproducible train/validation/test or benchmark-compatible split.
7. Apply model-specific scaling only using training information.
8. Save preprocessing configuration alongside experiment IDs.

## Known issues / risks

- The full benchmark is substantially larger than the MVP needs.
- The official repository notes that preprocessing the full experimental data can take hours on a standard PC.
- Mission 3 is omitted from ESA-ADB benchmarking for documented data-quality/benchmark reasons.
- Anomaly types can depend on the analyzed channel subset; if using subset-specific anomaly types, regenerate them according to the benchmark instructions.
- Rare nominal events may be treated as anomalies by some benchmark algorithms.

## What to watch out for

- Data leakage through future information.
- Random row splitting of time series.
- Normalization using test data.
- Treating anomaly labels as point labels when they represent intervals.
- Assuming channels are causally related.
- Claiming subsystem diagnosis without verified metadata.
- Calling anomaly scores "confidence" or "probability" without calibration.

## Fallback dataset: OPSSAT-AD (VALIDATED — 2026-08-15)

If ESA-ADB preprocessing is impractical for the hackathon, use OPSSAT-AD:
https://github.com/kplabs-pl/OPS-SAT-AD
Zenodo: https://doi.org/10.5281/zenodo.12588359

OPSSAT-AD is an ESA OPS-SAT telemetry anomaly benchmark with a much smaller footprint and published benchmark results.

### OPSSAT-AD Validation Results (2026-08-15)

**Files downloaded:**
- `data/raw/opssat-ad/dataset.csv` (496 KB) — segment-level statistical features
- `data/raw/opssat-ad/segments.csv` (17.5 MB) — raw telemetry time series

**Schema (segments.csv):**
| Column | Type | Description |
|--------|------|-------------|
| channel | object | Channel identifier (9 unique: CADC0872, CADC0892, CADC0874, CADC0884, CADC0873, CADC0886, CADC0888, CADC0894, CADC0890) |
| timestamp | datetime | ISO 8601 UTC timestamps |
| value | float64 | Telemetry value |
| label | object | Anomaly type: 'anomaly', 'a2', 'a3', 'a4' |
| sampling | int64 | Sampling rate (1 or 5 Hz) |
| anomaly | int64 | Binary anomaly flag (0=nominal, 1=anomaly) |
| segment | int64 | Segment ID (1-2123) |
| train | int64 | Train/test split flag (1=train, 0=test) |

**Schema (dataset.csv):**
| Column | Type | Description |
|--------|------|-------------|
| segment | int64 | Segment ID |
| anomaly | int64 | Segment-level anomaly label |
| train | int64 | Train/test split |
| channel | object | Channel identifier |
| sampling | int64 | Sampling rate |
| duration | int64 | Segment duration (seconds) |
| len | int64 | Number of points in segment |
| mean, var, std | float64 | Statistical moments |
| kurtosis, skew | float64 | Higher-order moments |
| n_peaks, smooth10_n_peaks, smooth20_n_peaks | int64 | Peak counts |
| diff_peaks, diff2_peaks | int64 | Difference peak counts |
| diff_var, diff2_var | float64 | Difference variances |
| gaps_squared, len_weighted, var_div_duration, var_div_len | float64 | Derived features |

**Key statistics:**
- Raw telemetry rows: 303,493
- Unique channels: 9
- Timestamp range: 2022-01-04 to 2022-06-02 (~149 days)
- Sampling rates: 1 Hz and 5 Hz
- Anomaly rows: 100,264 (33.04%)
- Anomaly segments: 434 out of 2,123 (20.4%)
- Train rows: 225,178 (74.2%)
- Test rows: 78,315 (25.8%)
- Segment features: 18 statistical features

**Anomaly representation:**
- Point-level labels in `segments.csv` (each row has anomaly flag)
- Segment-level labels in `dataset.csv` (each segment has single anomaly label)
- Anomaly types: 'anomaly', 'a2', 'a3', 'a4' (4 categories)
- Anomaly segment lengths: mean 231s, median 185s, max 1040s

**Critical findings for MissionGuard architecture:**
1. **Multiple channels available (9)** — but they appear in separate segments, not simultaneously aligned
2. **Timestamps NOT globally monotonic** — data is grouped by segment; segments from different channels/time periods are interleaved
3. **No channel metadata** — no subsystem/group mapping, no unit information, no channel descriptions
4. **Cross-channel aggregation NOT directly supported** — channels are not time-aligned; would need resampling/alignment assumptions
5. **Train/test split provided** — but not strictly temporal (segments from different times mixed in train/test)
6. **Anomaly types available** — could support multi-class detection but labels are segment-level

**Leakage risks:**
- Train/test split is by segment, not by time — future segments may appear in train
- Must verify temporal ordering if creating custom splits
- Segment features computed from entire segment (no leakage within segment)

**Suitability for MissionGuard workflow:**
| Workflow Step | Supported? | Notes |
|---------------|------------|-------|
| Telemetry ingestion | ✅ | CSV loadable |
| Data validation | ✅ | Schema validated |
| Preprocessing | ✅ | Features pre-computed; raw available |
| Anomaly detection | ✅ | Segment-level features or raw time series |
| Temporal incident aggregation | ⚠️ | Segments have timestamps; can group by time |
| Cross-channel aggregation | ❌ | Channels not time-aligned; no metadata |
| Priority scoring | ⚠️ | Can use segment features; limited evidence |
| Evidence packet | ⚠️ | Segment features + raw telemetry available |
| Granite briefing | ✅ | Structured evidence possible |

**Decision:** OPSSAT-AD is viable for Phases 0-3 (baseline detection, model experiments). For Phases 4+ (incident engine, cross-channel aggregation), ESA-ADB Mission 1 subset is preferred if downloadable. Will attempt ESA-ADB 3-month subset via TimeEval DatasetManager next.

## ESA-ADB Mission 1 — Next Steps

The ESA-ADB repo provides preprocessed data via TimeEval DatasetManager. The `datasets.csv` shows available configurations:

| Dataset | Train Size | Test Size | Dimensions | Contamination |
|---------|------------|-----------|------------|---------------|
| 3_months | 262,081 | 7,364,161 | 87 | 4.9% |
| 10_months | 878,401 | 7,364,161 | 87 | 2.2% |
| 21_months | 1,840,321 | 7,364,161 | 87 | 2.7% |
| 42_months | 3,677,761 | 7,364,161 | 87 | 1.8% |
| 84_months | 7,364,161 | 7,364,161 | 87 | 1.9% |

The 3_months subset (262K train rows, 87 channels) is the lightweight development target. Need to use TimeEval DatasetManager to download/access.

## Phase 1: Data Pipeline Implementation (2026-08-17)

### Reusable Components Created

**`src/missionguard/data/loaders.py`**
- `load_segments(path)` — Load raw telemetry with timestamp parsing and validation
- `load_dataset(path)` — Load segment features with validation
- `get_train_test_split(segments, dataset)` — Split using provided train/test column
- `load_opssat_ad(data_dir)` — Convenience loader for both files
- `get_temporal_train_test_split(df, test_ratio)` — Temporal split (no future leakage)

**`src/missionguard/data/schema.py`**
- `SegmentsSchema` / `DatasetSchema` — Dataclass schemas with required columns and dtypes
- `validate_segments_df(df)` — Returns `{"valid": bool, "errors": [], "warnings": []}`
- `validate_dataset_df(df)` — Same pattern for segment features
- Checks: required columns, dtypes, missing values, binary anomaly/train, positive segment IDs, sampling rates

**`src/missionguard/preprocessing/transforms.py`**
- `StandardScalerWrapper` / `RobustScalerWrapper` — Wrappers with feature names, fit/transform, save/load
- `fit_scaler(train_df, feature_names, scaler_type)` — Fit on train only
- `transform_features(df, scaler, feature_names)` — Apply fitted scaler
- `prepare_features_target(dataset_df, feature_names)` — Extract X, y
- `get_feature_names(df)` — Auto-detect feature columns (excludes metadata)

**`src/missionguard/preprocessing/time_series.py`**
- `sort_by_segment_time(df)` — Critical for OPSSAT-AD (segments concatenated)
- `extract_segment_windows(df, segment_id)` — Single segment time series
- `compute_rolling_features(series, windows, features)` — Rolling mean, std, min, max, skew, kurt
- `compute_differencing_features(series, lags)` — First and second order differences
- `detect_gaps(df, max_gap_seconds)` — Identify sampling irregularities
- `align_channels_temporally(df)` — Multi-channel alignment (requires time overlap)

**`src/missionguard/utils/config.py`**
- `OPSSAT_AD_CONFIG` — Complete dataset config with schemas, statistics, limitations
- `ESA_ADB_CONFIG` — Subset configurations for Mission 1
- `MODEL_CONFIG`, `EVAL_CONFIG` — Centralized hyperparameters

### Pipeline Usage Example

```python
from missionguard.data import load_opssat_ad, get_train_test_split
from missionguard.preprocessing import fit_scaler, transform_features, get_feature_names

# Load and validate
segments, dataset = load_opssat_ad("data/raw/opssat-ad")

# Split (using provided segment-based split)
train_seg, test_seg, train_ds, test_ds = get_train_test_split(segments, dataset)

# Or temporal split for raw time series
train_seg, test_seg = get_temporal_train_test_split(segments, test_ratio=0.25)

# Scale features (fit on train only!)
feature_names = get_feature_names(train_ds)
scaler = fit_scaler(train_ds, feature_names, scaler_type="robust")
train_scaled = transform_features(train_ds, scaler, feature_names)
test_scaled = transform_features(test_ds, scaler, feature_names)

# Prepare for modeling
X_train, y_train = prepare_features_target(train_scaled, feature_names)
X_test, y_test = prepare_features_target(test_scaled, feature_names)
```

### Tests

31 unit tests in `tests/test_data.py` and `tests/test_preprocessing.py`:
- Schema validation (missing columns, wrong dtypes, invalid values, missing data)
- Loader integration (real OPSSAT-AD data)
- Train/test splits (segment-based and temporal)
- Scaler fit/transform/persistence (StandardScaler, RobustScaler)
- Time series utilities (sorting, windowing, rolling features, differencing, gap detection)
- All tests pass