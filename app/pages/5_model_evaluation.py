"""
Model & Evaluation - Real production metrics and Phase 3b experiments
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import inject_global_styles, metric_row
from app.data_bridge import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODELS_DIR,
    PROJECT_ROOT,
    build_model_report,
    load_evaluation_metrics,
)

ARTIFACTS = PROJECT_ROOT / "artifacts" / "phase3b"


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_all():
    config = json.loads((DEFAULT_MODELS_DIR / "prod_config_v1.json").read_text())
    metrics = load_evaluation_metrics(DEFAULT_METRICS_PATH)
    fg = pd.read_csv(ARTIFACTS / "experiment_feature_groups.csv")
    cont = pd.read_csv(ARTIFACTS / "experiment_contamination_sweep.csv")
    nest = pd.read_csv(ARTIFACTS / "experiment_n_estimators_sweep.csv")
    return config, metrics, fg, cont, nest


def render():
    inject_global_styles()

    try:
        config, metrics, fg, cont, nest = load_all()
    except FileNotFoundError as exc:
        st.error(f"Experiment artifacts not found: {exc}")
        st.stop()

    # Header
    st.markdown(f"""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant); display: flex; justify-content: space-between; align-items: end; gap: 16px;">
        <div>
            <h1 style="margin: 0;">Model & Evaluation</h1>
            <div class="data-mono-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">
                {config['model_name']} v{config['version']} | trained {config.get('training_date', '')[:10]}
            </div>
        </div>
        <span class="mg-badge primary">{config['score_normalization']} scores | F1-optimal threshold</span>
    </div>
    """, unsafe_allow_html=True)

    # Real production metrics
    m = metrics or {}
    metric_row([
        {"title": "Precision", "value": f"{m['precision']:.1%}" if m else "N/A", "trend": f"{m['tp']} TP / {m['fp']} FP" if m else "", "variant": "primary", "icon": "target"},
        {"title": "Recall", "value": f"{m['recall']:.1%}" if m else "N/A", "trend": f"delay {m['mean_detection_delay_seconds']:.2f}s" if m else "", "variant": "warning", "icon": "search"},
        {"title": "F1 Score", "value": f"{m['f1']:.3f}" if m else "N/A", "trend": f"ROC-AUC {m['roc_auc']:.3f}" if m else "", "variant": "primary", "icon": "calculate"},
        {"title": "False Alarms", "value": f"{m['false_alarms_per_hour']:.0f}/hr" if m else "N/A", "trend": f"PR-AUC {m['pr_auc']:.3f}" if m else "", "variant": "error", "icon": "notifications"},
    ])

    col_left, col_right = st.columns([8, 4], gap="medium")

    with col_left:
        # Feature group ablation (the key Phase 3b finding)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fg["feature_group"],
            y=fg["f1"],
            marker_color=['#ffddb1' if f == fg["f1"].max() else '#a4e6ff' for f in fg["f1"]],
            opacity=0.85,
            hovertemplate='%{x}: F1=%{y:.3f}<extra></extra>',
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1c2023',
            plot_bgcolor='#0b0f11',
            font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
            margin=dict(l=60, r=20, t=30, b=40),
            title=dict(text="Feature group ablation — F1 by input features", font=dict(size=12)),
            yaxis=dict(title='F1', gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
            xaxis=dict(gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cont.iloc[:, 0], y=cont["f1"], mode='lines+markers',
            name='Contamination sweep', line=dict(color='#a4e6ff', width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=nest.iloc[:, 0], y=nest["f1"], mode='lines+markers',
            name='N_estimators sweep', line=dict(color='#ffddb1', width=2), yaxis='y2',
        ))
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1c2023',
            plot_bgcolor='#0b0f11',
            font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
            margin=dict(l=60, r=60, t=30, b=40),
            title=dict(text="Hyperparameter stability sweeps", font=dict(size=12)),
            xaxis=dict(title='Parameter value', gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
            yaxis=dict(title='Contamination sweep F1', gridcolor='#3c494e', side='left'),
            yaxis2=dict(title='N_estimators sweep F1', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation='h', yanchor='bottom', y=-0.35, xanchor='left', x=0),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        st.markdown("""
        <div class="mg-panel">
            <div class="mg-panel-header">
                <div class="mg-panel-title">
                    <span class="material-symbols-outlined" style="font-size: 18px;">dataset</span>
                    <span class="label-caps" style="letter-spacing: 0.1em;">Training Setup</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        split = config.get("data_split", {})
        setup_rows = pd.DataFrame([
            {"Setting": "Trained on", "Value": config.get("trained_on", "")},
            {"Setting": "Train segments", "Value": str(split.get("train_segments", ""))},
            {"Setting": "Test segments", "Value": str(split.get("test_segments", ""))},
            {"Setting": "Split strategy", "Value": split.get("split_strategy", "")},
            {"Setting": "Threshold", "Value": f"{config.get('threshold_value', 0):.4f}"},
            {"Setting": "Scaler", "Value": config.get("scaler_type", "")},
        ])
        st.dataframe(setup_rows, use_container_width=True, hide_index=True)

        report_bytes = json.dumps(build_model_report(), indent=2, default=str).encode("utf-8")
        st.download_button(
            "EXPORT LOG REPORT",
            data=report_bytes,
            file_name="missionguard_model_report.json",
            mime="application/json",
            use_container_width=True,
        )

    # Feature-group comparison table (native, sortable)
    st.markdown("<div class='label-caps' style='margin-top: 16px;'>Feature Group Experiments</div>", unsafe_allow_html=True)
    show_cols = ["feature_group", "n_features", "threshold", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    st.dataframe(
        fg[show_cols].sort_values("f1", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.set_page_config(
        page_title="MissionGuard - Model & Evaluation",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render()


if __name__ == "__main__":
    main()
