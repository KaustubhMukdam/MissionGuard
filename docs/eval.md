# Evaluation — MissionGuard

## Why these metrics

**Primary metric:** F1 score, supplemented by event-aware/operational metrics where the ESA-ADB evaluation pipeline supports them.

**Why:** Anomaly detection is typically imbalanced. Accuracy can be dominated by nominal telemetry and therefore should not be the primary decision metric.

**Secondary metrics:**
- Precision — controls the amount of false investigation work created by alerts.
- Recall — measures how many annotated anomalies are detected.
- PR-AUC — useful when anomalies are rare and ranking quality matters.
- False-alarm behavior — important for operator trust.
- Detection delay — important for operational usefulness where supported by the benchmark.
- Runtime/resource use — important for a deployable solo prototype.

## Baseline

At minimum compare against:
1. A simple statistical detector.
2. Isolation Forest or another lightweight unsupervised baseline.
3. A temporal/deep model only if justified by the data and compute budget.

The final model must beat the baseline meaningfully on the chosen evaluation protocol, not merely on training data.

## Results vs baseline

| Model | Precision | Recall | F1 | PR-AUC | False alarms | Detection delay | Runtime |
|-------|-----------|--------|----|--------|--------------|-----------------|---------|
| Statistical baseline (MAD) | 0.735 | 0.319 | 0.444 | 0.620 | ~1,300/hr | 0.1s | <1s |
| Isolation Forest (all 18 features) | 0.299 | 0.805 | 0.437 | 0.376 | ~1,450/hr | 0.1s | <1s |
| **Isolation Forest (peak features only)** | **0.781** | **0.566** | **0.656** | **0.630** | **~850/hr** | **0.1s** | <1s |
| Rolling MAD (60s window) | 0.285 | 0.058 | 0.096 | TBD | TBD | TBD | <1s |
| Rolling Z-Score (60s window) | 0.349 | 0.057 | 0.098 | TBD | TBD | TBD | <1s |

**Best single model:** Isolation Forest with peak-based features (3 features: n_peaks, smooth10_n_peaks, smooth20_n_peaks) — F1=0.656

## Evaluation protocol

- Use a time-aware split compatible with the benchmark where possible.
- Never randomly mix future telemetry into training.
- Keep preprocessing fit parameters confined to training data.
- Evaluate at both point/segment level and event level when meaningful.
- Preserve the benchmark's anomaly interval definitions.
- Report the exact channels, mission, split, model version, and threshold.
- Record threshold selection separately from test evaluation.

## Error analysis

For false positives:
- Was the signal genuinely unusual but unlabeled?
- Did a normal operating-mode change trigger the detector?
- Did missing/gapped data create an artifact?
- Is the threshold too sensitive?

For false negatives:
- Was the anomaly too short?
- Was it masked by noisy telemetry?
- Did the model fail on a particular channel/group?
- Did preprocessing remove useful signal?

## Incident-level evaluation (Phase 4)

| Metric | Description | Target |
|--------|-------------|--------|
| Incident precision | % of incidents that correspond to true anomaly clusters | > 0.7 |
| Incident recall | % of true anomaly clusters captured as incidents | > 0.7 |
| Incident F1 | Harmonic mean of incident precision/recall | > 0.7 |
| Temporal alignment | Mean temporal offset between incident and true cluster | < 5 min |
| Channel coverage | % of affected channels captured per incident | > 0.8 |
| Priority ranking quality | % of high-priority incidents that are true anomalies | > 0.8 |

## What the numbers actually mean

A recall of 0.80 means the detector identified 80% of the evaluated positive anomaly instances under the specified evaluation protocol. It does not mean the spacecraft is 80% safe, nor does it mean there is an 80% probability of failure.

A precision of 0.70 means 70% of the evaluated flagged instances were positive under the benchmark definition. It does not mean 70% of alerts are physical failures in real operations.

## Product-level evaluation

MissionGuard should also be evaluated as a system:

- Can an operator identify the highest-priority incident quickly?
- Can every AI claim be traced to evidence?
- Does incident grouping reduce alert clutter?
- Does the priority score rank meaningful incidents above isolated low-value anomalies?
- How long does it take from telemetry input to Incident Autopsy?

## Incident Engine evaluation metrics

| Component | Metric | Current | Target |
|-----------|--------|---------|--------|
| Event detection | Point-level F1 | 0.656 (peak features) | > 0.7 |
| Temporal aggregation | Incident recall | TBD | > 0.7 |
| Temporal aggregation | Temporal alignment (mean offset) | < 1 min | < 5 min |
| Priority scoring | Ranking quality (NDCG@5) | TBD | > 0.8 |
| Evidence packets | Completeness (all required fields) | 100% | 100% |
| LLM briefing | Groundedness (no hallucinated telemetry) | 100% (template) | 100% |
| End-to-end latency | Telemetry → Incident Autopsy | < 1s | < 5s |