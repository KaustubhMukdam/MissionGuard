# Tasks — MissionGuard

## In progress

- [ ] Implement statistical baseline
- [ ] Implement Isolation Forest baseline
- [ ] Define anomaly-event conversion rules
- [ ] Define temporal incident aggregation
- [ ] Define priority-score formula
- [ ] Build first Incident Autopsy prototype
- [ ] Evaluate whether LSTM/autoencoder adds value
- [ ] Build evidence packet schema
- [ ] Integrate Granite/watsonx only after the analytical pipeline works
- [ ] Create Streamlit dashboard
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

## Blocked

- [ ] Exact model choice — blocked until baseline experiments are complete
- [ ] Exact Granite model/API — blocked until account access and free-tier availability are verified
- [ ] Exact persistence design — blocked until the need for incident history is confirmed
- [ ] ESA-ADB Mission 1 3-month subset download — blocked on TimeEval DatasetManager access

## Ideas / backlog

- [ ] Mission 2 generalization test
- [ ] Historical incident similarity
- [ ] Documentation RAG
- [ ] Real-time telemetry replay
- [ ] Exportable incident report
- [ ] Advanced event-level metrics
- [ ] Public benchmark comparison page