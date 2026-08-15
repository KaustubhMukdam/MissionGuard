# Tech Stack — MissionGuard

## Frontend

| Technology | Version | Why chosen |
|------------|---------|------------|
| Streamlit | Pin during implementation | Fast Python-native dashboard; minimizes frontend overhead for a solo build |
| Plotly | Pin during implementation | Interactive telemetry and incident visualizations |

## Backend

| Technology | Version | Why chosen |
|------------|---------|------------|
| Python | 3.11+ target; validate against dependencies | Strong time-series/ML ecosystem and user familiarity |
| FastAPI | Optional | Use only if a clean API boundary becomes necessary after the MVP works |

## Database / Storage

| Technology | Why chosen |
|------------|------------|
| None for MVP | Avoid infrastructure until persistence is actually needed |
| SQLite (optional) | Simple local persistence for incident history without a hosted database |

## ML/AI

| Library / Service | Why chosen |
|-------------------|------------|
| NumPy | Numerical operations |
| pandas | Telemetry loading, cleaning, time-indexed manipulation |
| scikit-learn | Baselines and lightweight anomaly detection |
| TensorFlow/Keras | Candidate for LSTM/autoencoder experiments only if justified by results |
| IBM watsonx.ai / Granite | Grounded explanation layer and potential IBM-native time-series experimentation; not a hard dependency until account/model access is verified |

## Data

| Dataset | Role |
|---------|------|
| ESA Anomaly Dataset / ESA-ADB | Primary benchmark: real satellite telemetry from three ESA missions with curated anomaly annotations; ESA-ADB benchmarks two missions |
| OPSSAT-AD | Fallback dataset if ESA-ADB preprocessing becomes impractical |

## Deployment

| Technology | Why chosen |
|------------|------------|
| Streamlit Community Cloud | Simple public demo path if the final app fits its deployment constraints |
| GitHub | Source control and public hackathon repository |

## Alternatives considered

- React + TypeScript: rejected for MVP because it adds frontend/API overhead without improving the core ML demonstration.
- PostgreSQL/Supabase: rejected until persistence/auth is actually required.
- Docker/Kubernetes: rejected for MVP scope.
- Large hosted GPU: rejected because the initial benchmark subset and candidate models should be laptop-friendly.
- RAG/vector database: postponed until a real documentation-retrieval requirement exists.
- OPSSAT-AD as primary: rejected provisionally because ESA-ADB better supports multivariate/operational incident reasoning.

## Known tradeoffs

- Streamlit is faster to build than a separate frontend but gives less control over complex UI architecture.
- Local ML keeps costs low but limits model scale.
- Granite/watsonx integration may have account/model availability and usage limits; the product must remain functional without paid inference.
- ESA-ADB is more operationally relevant than small datasets but requires more careful preprocessing and resource management.
