# MissionGuard

AI-assisted spacecraft telemetry intelligence for anomaly detection, incident aggregation, prioritization, and evidence-grounded operator briefings.

## Current status

**Ideation finalized — implementation not started.**

Primary benchmark direction: **ESA Anomaly Dataset / ESA-ADB**, starting with a lightweight Mission 1 subset. OPSSAT-AD is retained as a fallback.

## MVP

```text
ESA telemetry
    ↓
validation + preprocessing
    ↓
anomaly detection
    ↓
temporal / cross-channel incident aggregation
    ↓
priority scoring
    ↓
evidence packet
    ↓
Granite / LLM briefing
    ↓
Incident Autopsy
```

## Documentation

Start with:

1. `docs/project_context.md`
2. `docs/PRD.md`
3. `docs/data_doc.md`
4. `docs/architecture.md`
5. `docs/tasks.md`
6. `docs/experiment_log.md`

## Important scope rule

MissionGuard is a human-in-the-loop research/hackathon prototype. It does not autonomously command spacecraft and does not claim definitive physical root-cause diagnosis.

## Primary references

- ESA dataset: https://github.com/esa/anomaly-dataset
- ESA-ADB benchmark: https://github.com/kplabs-pl/ESA-ADB
- ESA-AD Zenodo: https://doi.org/10.5281/zenodo.12528696
- OPSSAT-AD fallback: https://github.com/kplabs-pl/OPS-SAT-AD
