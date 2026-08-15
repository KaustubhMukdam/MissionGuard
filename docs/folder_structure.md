# Folder Structure — MissionGuard

```text
MissionGuard/
├── docs/
│   ├── project_context.md
│   ├── PRD.md
│   ├── tech_stack.md
│   ├── architecture.md
│   ├── folder_structure.md
│   ├── tasks.md
│   ├── design_prompt.md
│   ├── learnings.md
│   ├── debug_log.md
│   ├── experiment_log.md
│   ├── model_card.md
│   ├── data_doc.md
│   └── eval.md
│
├── data/
│   ├── raw/                 # Downloaded source data; do not commit large files
│   ├── interim/             # Cleaned/intermediate artifacts
│   └── processed/           # Model-ready subsets
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_experiments.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   └── missionguard/
│       ├── data/            # Loaders and validation
│       ├── preprocessing/   # Time-series preprocessing
│       ├── models/          # Baselines and ML models
│       ├── detection/       # Score-to-event conversion
│       ├── incidents/       # Temporal/group aggregation + priority
│       ├── explanation/     # Evidence packet + LLM adapter
│       ├── evaluation/      # Metrics and evaluation utilities
│       └── utils/           # Shared helpers/configuration
│
├── app/
│   └── streamlit_app.py     # MVP dashboard entry point
│
├── tests/
│   ├── test_data.py
│   ├── test_detection.py
│   ├── test_incidents.py
│   └── test_explanation.py
│
├── models/                  # Small model artifacts only; large files ignored
├── artifacts/               # Evaluation outputs and plots
├── requirements.txt
├── .gitignore
└── README.md
```

## Naming conventions

- Python files: `snake_case.py`
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Notebook names: numbered by workflow stage
- Experiment IDs: `EXP-001`, `EXP-002`, ...
- Incident IDs: `INC-001`, `INC-002`, ...
- Model artifacts: include model name + version + experiment ID

## Structure rule

Keep domain logic outside Streamlit. The dashboard should call reusable functions from `src/missionguard/`; notebooks are for exploration, not the final application logic.
