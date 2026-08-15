# PRD — MissionGuard

## Problem statement

Spacecraft generate large volumes of telemetry. Operators need to identify abnormal behavior quickly, distinguish meaningful incidents from isolated alerts, understand the evidence behind an alert, and decide what deserves investigation.

MissionGuard converts raw spacecraft telemetry into prioritized, evidence-backed incidents and an AI-generated investigation briefing.

The system is intended as a hackathon prototype and research-oriented decision-support tool, not an operationally certified flight system.

## Target users

Primary:
- Spacecraft operations engineers
- Satellite telemetry analysts
- AI/ML researchers working on spacecraft anomaly detection

Secondary:
- Students and researchers demonstrating operational AI
- Hackathon judges evaluating AI + space operations applications

## Core features (MVP)

- [ ] **Telemetry ingestion and validation** — Load the selected ESA-ADB subset, validate timestamps/channels/labels, and report data-quality issues.
- [ ] **Anomaly detection** — Produce an anomaly score using a transparent baseline and at least one ML approach.
- [ ] **Temporal incident aggregation** — Group temporally related anomalous events into a single incident.
- [ ] **Cross-channel evidence aggregation** — When supported by the selected subset, identify channels/groups participating in the same incident window.
- [ ] **Priority scoring** — Produce a transparent operational prioritization score from measurable evidence such as anomaly magnitude, duration, recurrence, and affected-channel count.
- [ ] **Incident Autopsy** — Show what happened, why it was flagged, supporting telemetry, affected signals/groups, priority, and what an operator should investigate.
- [ ] **Grounded AI briefing** — Send structured evidence to an IBM Granite/LLM layer and generate a concise explanation without allowing the model to invent telemetry facts.
- [ ] **Dashboard** — Provide mission overview, telemetry visualization, active incidents, and incident detail views.

## Nice-to-have (post-MVP)

- [ ] Mission 2 cross-mission validation
- [ ] Historical incident similarity
- [ ] RAG over approved spacecraft documentation
- [ ] Real-time telemetry replay
- [ ] More advanced multivariate models
- [ ] Persistent incident history
- [ ] API separation with FastAPI
- [ ] Exportable incident report

## Non-goals

- Autonomous spacecraft commands
- Certified flight/ground operations
- Physical root-cause certainty
- Causal inference between telemetry channels
- Multi-agent orchestration
- Kubernetes/microservice infrastructure
- Mandatory GPU deployment
- Full ESA-ADB processing for the MVP
- RAG in the first working version
- React frontend unless Streamlit proves insufficient

## Success metrics

Technical:
- Beat a simple baseline on the selected evaluation metric(s).
- Report precision, recall, F1 and appropriate event/operational metrics.
- Quantify false alarms and detection delay where the benchmark supports them.
- Produce reproducible experiments from documented data splits.

Product:
- A judge can select a telemetry window and understand why MissionGuard created an incident.
- Every AI-generated statement can be traced to structured evidence.
- Incident Autopsy can be demonstrated end-to-end in under three minutes.

## Constraints

- Time: Hackathon prototype; prioritize a reliable MVP over breadth.
- Cost: Target ₹0; use free/open-source components and free allowances where available.
- Tech: Python-first.
- Development: IBM Bob must be the primary AI development tool.
- Hardware: Must remain viable on a normal student laptop without requiring a GPU.
- Scientific integrity: Never claim capabilities not supported by the selected dataset or evaluation.
