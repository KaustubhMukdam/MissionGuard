# Tasks — MissionGuard

## In progress

- [ ] Build first Incident Autopsy prototype
- [ ] Evaluate whether LSTM/autoencoder adds value
- [ ] Integrate Granite/watsonx only after the analytical pipeline works
- [ ] Create Streamlit dashboard
  - [x] Page 1 (Mission Overview) wired to live pipeline
  - [ ] Pages 2–5 still consume mock data — next: Incident Center + Autopsy via `run_pipeline()` output
- [ ] Add unit tests for non-trivial functions
- [ ] Prepare public deployment

## Done

- [x] Finalize MissionGuard product direction
- [x] Reject unsupported "root-cause certainty" claims
- [x] Replace "91% confidence" language with "anomaly score"
- [x] Decide against RAG/agents/microservices for MVP
- [x] Select ESA-ADB as primary dataset direction
- [x] Keep OPSSAT-AD as fallback
- [x] Validate ESA-ADB access and download required development subset
- [x] Inspect Mission 1 telemetry schema and channel metadata (via OPSSAT-AD fallback)
- [x] Confirm labels/anomaly event structure
- [x] Produce the first 5+ telemetry plots
- [x] Record data quality findings in `data_doc.md`
- [x] Create deterministic train/validation/test strategy
- [x] Create reusable data loading (`src/missionguard/data/loaders.py`)
- [x] Create schema validation (`src/missionguard/data/schema.py`)
- [x] Create preprocessing transforms (`src/missionguard/preprocessing/transforms.py`)
- [x] Create time series utilities (`src/missionguard/preprocessing/time_series.py`)
- [x] Add unit tests for data loading, validation, preprocessing (31 tests passing)
- [x] Implement statistical baseline (MAD, Z-score, Rolling variants)
- [x] Implement Isolation Forest baseline
- [x] Create base anomaly detector class with threshold tuning
- [x] Create score-to-event conversion (`src/missionguard/detection/events.py`)
- [x] Create event merging and filtering
- [x] Create threshold selection and evaluation (`src/missionguard/detection/thresholding.py`)
- [x] Create evaluation metrics (`src/missionguard/evaluation/metrics.py`)
- [x] Create experiment runner (`src/missionguard/evaluation/experiment.py`)
- [x] Add unit tests for models, detection, evaluation (85 tests passing)
- [x] Phase 3a: Rolling baselines on raw telemetry (OPSSAT-AD)
  - Rolling MAD/Z-Score baselines on 18 rolling features (windows: 10s, 30s, 60s)
  - Event detection from rolling scores
  - Temporal incident aggregation (merge_events with configurable gap)
- [x] Phase 3b: Isolation Forest Production Baseline + Extended Experiments
  - Production model saved: `models/isolation_forest_prod_v1.joblib`
  - Scaler saved: `models/robust_scaler_prod_v1.joblib`
  - Config saved: `models/prod_config_v1.json`
  - F1=0.4365 (all 18 features) → **F1=0.656 (peak-based 3 features only!)**
  - Contamination sweep: stable across 0.01-0.2
  - N_estimators: stable 50-500, peak at 300
  - Peak-based features (3): n_peaks, smooth10_n_peaks, smooth20_n_peaks
  - Normalization method doesn't matter with F1-optimal threshold
  - Bootstrap 95% CI: F1 ∈ [0.390, 0.480]
- [x] Phase 4: Incident Engine (Priority scoring, evidence packets, Incident Autopsy)
  - Temporal incident aggregation (`src/missionguard/incidents/aggregation.py`)
  - Priority scoring with configurable weights (`src/missionguard/incidents/priority.py`)
  - Evidence packet schema (`src/missionguard/incidents/evidence.py`)
  - LLM briefing template for grounded briefings
  - Unit tests for aggregation, priority, evidence (20 tests passing)
  - Total tests: 105 passing
- [x] Phase 5a: Backend↔Frontend bridge (`app/data_bridge.py`, 2026-08-25)
  - `load_production_models()` / `load_evaluation_metrics()` — artifact loading with graceful metrics fallback
  - `run_pipeline()` — prod artifacts → load OPSSAT-AD → segment-based split → scale → IsolationForest score → per-channel events (`get_events_per_channel`) → incidents (5-min gap) → priority ranking → `EvidencePacket` per incident
  - Per-segment time windows joined from raw segments.csv (dataset.csv carries no timestamps)
  - Non-finite feature rows dropped BEFORE scaling/scoring (sklearn raises on NaN/inf; loader validation only warns on inf)
  - `build_dashboard_view()` — pure function mapping pipeline result to the exact dict Mission Overview renders (honest KPIs: incident count / event count / model F1 / segments scored)
  - `1_mission_overview.py` wired to live pipeline via `@st.cache_resource`; FileNotFoundError guard for missing artifacts; trend chart shows real CADC0872 telemetry with anomaly markers
  - Tests: `tests/test_bridge.py` — hermetic synthetic workspace fixture (tmp_path), 16 tests covering normal + edge cases (empty test split, missing model file, inf rows, threshold respect, ranking order, packet validation)
  - Total tests: 121 passing

## Blocked

- [ ] Exact Granite model/API — blocked until account access and free-tier availability are verified
- [ ] Exact persistence design — blocked until the need for incident history is confirmed
- [ ] ESA-ADB Mission 1 3-month subset download — blocked on TimeEval DatasetManager access (not in GitHub, requires 3.7GB download + hours preprocessing)

## Ideas / backlog

- [ ] Mission 2 generalization test
- [ ] Historical incident similarity
- [ ] Documentation RAG
- [ ] Real-time telemetry replay
- [ ] Exportable incident report
- [ ] Advanced event-level metrics
- [ ] Public benchmark comparison page