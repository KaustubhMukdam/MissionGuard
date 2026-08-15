# Model Card — MissionGuard Anomaly Detection Model

## Model overview

- **Model name:** TBD after baseline experiments
- **Type:** TBD; candidates include statistical baseline, Isolation Forest, and temporal neural model
- **Task:** Spacecraft telemetry anomaly detection
- **Training date:** TBD
- **Framework:** TBD

## Training data

- **Source:** ESA Anomaly Dataset / ESA-ADB
- **Initial subset:** Lightweight Mission 1 subset; exact channels and rows to be recorded after local validation
- **Features used:** TBD after schema inspection
- **Target:** Anomaly event/segment labels supplied by the benchmark
- **Preprocessing:** TBD and documented in `data_doc.md`

## Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Precision | TBD | TBD | TBD |
| Recall | TBD | TBD | TBD |
| F1 | TBD | TBD | TBD |
| PR-AUC | TBD | TBD | TBD |
| False-alarm metric | TBD | TBD | TBD |
| Detection delay | TBD | TBD | TBD |

## What it does well

TBD after evaluation. Do not write claims before measuring them.

## Known limitations

- The model is a research/hackathon prototype.
- An anomaly score is not automatically a probability of physical failure.
- Dataset performance does not establish flight-readiness.
- Model performance may vary across missions, channels, operating modes, and data-quality conditions.
- A detected anomaly does not prove physical root cause.

## Bias and fairness

Traditional demographic fairness metrics are not the primary concern for this telemetry task.

Relevant reliability checks:
- [ ] Performance by channel/group
- [ ] Performance by anomaly type
- [ ] Performance by mission phase where supported
- [ ] False alarms during nominal operating periods
- [ ] Sensitivity to missing/irregular telemetry

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
