# Model Card — MissionGuard Anomaly Detection Model

## Model overview

- **Model name:** IsolationForestDetector (Production v1.0)
- **Type:** Isolation Forest (unsupervised ensemble)
- **Task:** Spacecraft telemetry anomaly detection (segment-level)
- **Training date:** 2026-08-17
- **Framework:** scikit-learn 1.5.0
- **Experiment ID:** EXP-005

## Training data

- **Source:** ESA Anomaly Dataset / ESA-ADB (OPSSAT-AD fallback)
- **Subset:** OPSSAT-AD segment features (dataset.csv)
- **Samples:** 1,594 train / 529 test segments
- **Features used:** 18 segment-level statistical features (duration, len, mean, var, std, kurtosis, skew, n_peaks, smooth10_n_peaks, smooth20_n_peaks, diff_peaks, diff2_peaks, diff_var, diff2_var, gaps_squared, len_weighted, var_div_duration, var_div_len)
- **Target:** Segment-level anomaly labels (binary: 0=nominal, 1=anomaly)
- **Anomaly rate:** ~21% in train, ~21% in test
- **Preprocessing:** RobustScaler fitted on training data only (no leakage)

## Performance (Production Baseline - All 18 Features)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Precision | — | 0.299 | 0.299 |
| Recall | — | 0.805 | 0.805 |
| F1 | — | 0.437 | 0.437 |
| PR-AUC | — | 0.376 | 0.376 |
| ROC-AUC | — | 0.636 | 0.636 |
| False-alarm rate | — | ~1,450/hr | ~1,450/hr |
| Detection delay | — | 0.1s | 0.1s |

**Threshold:** F1-optimal on validation (0.108)
**Contamination:** 0.1 (fixed)
**N_estimators:** 200
**Score normalization:** minmax

## Performance (Optimized - Peak-Based 3 Features Only)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Precision | — | 0.781 | 0.781 |
| Recall | — | 0.566 | 0.566 |
| F1 | — | 0.656 | **0.656** |
| PR-AUC | — | 0.630 | 0.630 |

**Features used:** `n_peaks`, `smooth10_n_peaks`, `smooth20_n_peaks` (3 features)
**Improvement:** +50% F1 over all 18 features

## What it does well

- High recall on segment-level anomalies (catches 80% of anomalies with all features, 57% with peak features)
- Stable across contamination values (0.01-0.2) and n_estimators (50-500)
- Score normalization method doesn't affect F1 when using F1-optimal threshold
- Bootstrap 95% CI for F1: [0.390, 0.480] (stable)

## Known limitations

- The model is a research/hackathon prototype.
- An anomaly score is not automatically a probability of physical failure.
- Dataset performance does not establish flight-readiness.
- Model performance may vary across missions, channels, operating modes, and data-quality conditions.
- A detected anomaly does not prove physical root cause.
- **High false alarm rate (~1,450/hr)** with all features — operationally unacceptable without temporal filtering
- Peak-feature model has lower recall (57%) but much higher precision (78%)
- OPSSAT-AD segments are pre-extracted around events — limits generalization
- No cross-channel temporal correlation (single channel per segment)

## Bias and fairness

Traditional demographic fairness metrics are not the primary concern for this telemetry task.

Relevant reliability checks:
- [x] Performance by channel/group (single-channel only)
- [ ] Performance by anomaly type (a2, a3, a4)
- [ ] Performance by mission phase where supported
- [ ] False alarms during nominal operating periods
- [x] Sensitivity to missing/irregular telemetry

## Intended use

- Research and hackathon demonstration
- Spacecraft telemetry anomaly detection experiments
- Human-in-the-loop incident triage prototype

## Out-of-scope use

- Autonomous spacecraft control
- Safety-critical flight decisions
- Unsupervised issuance of spacecraft commands
- Certifying hardware/software health
- Treating model output as definitive physical diagnosis

## Production artifacts

- Model: `models/isolation_forest_prod_v1.joblib`
- Scaler: `models/robust_scaler_prod_v1.joblib`
- Config: `models/prod_config_v1.json`
- Experiment data: `artifacts/phase3b/`