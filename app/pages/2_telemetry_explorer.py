"""
Telemetry Explorer - Interactive telemetry visualization (live pipeline data)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import inject_global_styles
from app.data_bridge import run_pipeline, telemetry_slice


@st.cache_resource(show_spinner="Running MissionGuard detection pipeline...")
def load_pipeline():
    return run_pipeline()


def render():
    inject_global_styles()

    result = load_pipeline()
    scored = result["scored"]
    threshold = result["model_info"].get("threshold")

    if scored.empty:
        st.info("No scored telemetry available. Check that the OPSSAT-AD dataset and production model artifacts exist.")
        st.stop()

    channel = str(scored["channel"].iloc[0])
    max_score = float(scored["anomaly_score"].max())
    anomaly_count = int((scored["anomaly_score"] >= threshold).sum()) if threshold is not None else 0

    # Header
    st.markdown(f"""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant);">
        <h1 style="margin: 0;">Telemetry Explorer</h1>
        <div class="body-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">
            Channel {channel} | {len(scored)} segments | {result['model_info']['name']} v{result['model_info']['version']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: inspection panel from real evidence
    with st.sidebar:
        st.markdown("<div class='label-caps' style='margin: 16px 0 8px;'>Inspection Details</div>", unsafe_allow_html=True)
        latest = scored.iloc[-1]
        st.metric("Current Value", f"{float(latest['value']):.3e}")

        severity_variant = "error" if max_score >= 0.75 else "warning" if max_score >= 0.5 else "primary"
        st.markdown(f"""
        <div class="mg-flex mg-gap-sm" style="align-items: center;">
            <span class="mg-badge {severity_variant}">Max score {max_score:.2f}</span>
            <span class="mg-badge nominal">{anomaly_count} anomalies</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.caption(
            f"Threshold: {threshold:.3f}\n\n"
            f"Sampling rate: {int(scored['sampling'].iloc[0])} Hz\n\n"
            f"Window: {scored['timestamp'].min():%Y-%m-%d %H:%M:%S} → {scored['timestamp'].max():%Y-%m-%d %H:%M:%S}"
        )

    # Time range slider over scored segments
    start_idx, end_idx = st.slider(
        "Time Range",
        0, len(scored),
        (max(0, len(scored) - 300), len(scored)),
        format="%d segments",
        label_visibility="collapsed",
    )

    sl = telemetry_slice(scored, start_idx, end_idx, threshold)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sl["timestamps"],
        y=sl["values"],
        mode='lines',
        name='Observed',
        line=dict(color='#a4e6ff', width=1.5),
    ))

    expected = pd.Series(sl["values"]).rolling(window=20, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=sl["timestamps"],
        y=expected,
        mode='lines',
        name='Rolling mean',
        line=dict(color='#bbc9cf', width=1.5, dash='dash'),
    ))

    if sl["anomaly_positions"]:
        pos = sl["anomaly_positions"]
        fig.add_trace(go.Scatter(
            x=[sl["timestamps"][p] for p in pos],
            y=[sl["values"][p] for p in pos],
            mode='markers',
            name=f'Anomaly (≥{threshold:.2f})' if threshold else 'Anomaly',
            marker=dict(color='#ffb4ab', size=7, symbol='x'),
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#101416',
        plot_bgcolor='#0b0f11',
        font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
        margin=dict(l=60, r=60, t=10, b=40),
        xaxis=dict(gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
        yaxis=dict(title='Mean value / segment', gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(family='JetBrains Mono', size=10)),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Working export of the visible slice
    export_df = scored.iloc[start_idx:end_idx][["segment", "timestamp", "value", "anomaly_score"]]
    st.download_button(
        "EXPORT CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"telemetry_{channel}_{start_idx}_{end_idx}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    st.set_page_config(
        page_title="MissionGuard - Telemetry Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render()


if __name__ == "__main__":
    main()
