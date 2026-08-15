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
| Statistical baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Isolation Forest | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LSTM/Autoencoder (if used) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Final model** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

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
