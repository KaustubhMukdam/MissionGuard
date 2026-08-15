# Architecture — MissionGuard

## System overview

MissionGuard is a Python-first anomaly intelligence pipeline. ESA telemetry is validated and transformed into model-ready time series. A baseline and one or more ML detectors produce anomaly scores. The incident engine groups temporally and, where justified by the dataset, cross-channel related anomalies. A transparent priority score is calculated from measurable evidence. The resulting structured incident packet is shown directly in the dashboard and may also be passed to IBM Granite/another approved LLM to produce a concise operator briefing.

The LLM is downstream of the analytical system. It does not determine whether telemetry is anomalous and is not allowed to invent telemetry facts.

## Component diagram

```text
                    [ESA-ADB Telemetry]
                             |
                             v
                    [Data Validation]
                             |
                             v
                    [Preprocessing]
                             |
              +--------------+--------------+
              |                             |
              v                             v
      [Statistical Baseline]        [ML Anomaly Model]
              |                             |
              +--------------+--------------+
                             |
                             v
                      [Anomaly Scores]
                             |
                             v
                 [Incident Aggregation]
                    /                 \
                   /                   \
          Temporal grouping       Cross-channel/group
                                  evidence when valid
                   \                   /
                    \                 /
                     +-------+-------+
                             |
                             v
                    [Priority Scoring]
                             |
                             v
                     [Evidence Packet]
                       /           \
                      /             \
                     v               v
             [Streamlit UI]    [Granite / LLM]
                                      |
                                      v
                              [Grounded Briefing]
                                      |
                                      v
                               [Incident Autopsy]
```

## Data flow

1. Load the selected ESA-ADB subset.
2. Validate timestamps, channel identifiers, missingness, labels, and ordering.
3. Apply only preprocessing justified by the dataset and model.
4. Create a baseline detector.
5. Train/evaluate candidate anomaly detectors using a documented split.
6. Convert point/segment scores into anomaly events.
7. Group events by configurable temporal windows.
8. Aggregate cross-channel evidence only when channel relationships are supported by the dataset.
9. Calculate a transparent priority score.
10. Build a structured incident packet.
11. Render telemetry, evidence, metrics, and incident details in Streamlit.
12. Optionally send the structured packet to Granite/LLM for a grounded operator briefing.
13. Display the briefing beside the underlying evidence.

## Key interfaces

- Data layer → ML layer: pandas DataFrames / typed internal records.
- ML layer → incident engine: anomaly events with timestamps, channel IDs, scores, and model metadata.
- Incident engine → UI: structured incident objects.
- Incident engine → LLM: evidence-only JSON/structured prompt.
- UI → optional API: no separate API in MVP; FastAPI may be introduced later if required.

## Evidence contract

Every generated briefing must receive:
- incident ID
- time window
- affected channels/groups
- anomaly scores
- baseline/model used
- observed vs expected summaries
- duration
- recurrence information
- priority score and its components
- evaluation/model version

The LLM must not receive a blank prompt such as “tell me what happened.” It must reason from the evidence packet.

## Security considerations

- [ ] No API keys in source code.
- [ ] Environment variables for IBM credentials.
- [ ] Validate uploaded/input files.
- [ ] Restrict file size and supported formats.
- [ ] Treat telemetry as data, not executable content.
- [ ] Never expose credentials in logs.
- [ ] Make clear that AI output is decision support, not an autonomous command.

## Reliability considerations

- Store model version and experiment ID with each incident.
- Make anomaly thresholds configurable and documented.
- Provide a deterministic fallback explanation if the LLM is unavailable.
- Keep the ML pipeline runnable without cloud services.
