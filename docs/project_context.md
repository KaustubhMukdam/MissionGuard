# MissionGuard — Project Context

## Project
Name: MissionGuard
Status: Pre-development / ideation finalized
Started: 2026-08-12

## One-liner
MissionGuard is an AI-assisted spacecraft telemetry intelligence system that detects anomalous behavior, groups related events into operational incidents, prioritizes them, and generates evidence-grounded investigation briefings for human operators.

## Stack
- Frontend: Streamlit
- Backend: Python; FastAPI only if a separate API becomes necessary
- Database: None for MVP; SQLite only if persistence is required
- ML: pandas, NumPy, scikit-learn, TensorFlow/Keras if justified by experiments
- AI: IBM watsonx.ai / Granite as an explanation layer when accessible
- Data: ESA Anomaly Dataset / ESA-ADB lightweight subset
- Deployment: Streamlit Community Cloud if practical
- Development: IBM Bob as the primary AI development tool

## Key decisions made
- ESA-ADB is the primary benchmark because it is real ESA telemetry with curated annotations and an operationally oriented benchmark.
- Start with a lightweight subset rather than the full dataset.
- Core anomaly detection must be measurable and independent of the LLM.
- Use temporal/cross-channel incident aggregation rather than claiming causal root-cause analysis.
- Use “anomaly score” unless a probability/confidence measure is explicitly calibrated.
- Incident Autopsy is the product centerpiece.
- No RAG, agents, microservices, React, GPU requirement, or autonomous spacecraft commands in MVP.

## Current focus
Validate the ESA-ADB subset, reproduce/establish a baseline, and confirm the data schema before building the dashboard.

## Known issues / blockers
- Exact subset/schema and channel metadata must be validated locally before implementation decisions are frozen.
- IBM model availability and free-tier limits must be checked against the actual account before making Granite a hard dependency.

## What this project is NOT doing
MissionGuard is not a certified spacecraft operations system, autonomous command system, causal diagnosis engine, or replacement for human operators.
