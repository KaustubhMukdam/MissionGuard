"""
Incident Autopsy - Deep dive investigation view (live evidence packets, native UI)
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import inject_global_styles
from app.data_bridge import run_pipeline, event_window_series, briefing_from_packet


@st.cache_resource(show_spinner="Running MissionGuard detection pipeline...")
def load_pipeline():
    return run_pipeline()


def resolve_packet(result):
    """Pick packet from ?id= query param (deep link), else the top-ranked incident."""
    packets = result["packets"]
    if not packets:
        return None
    requested = st.query_params.get("id")
    if requested and requested in packets:
        return requested, packets[requested]
    return next(iter(packets.items()))


def sync_query_param(incident_id: str) -> None:
    """Keep ?id= in sync without triggering a Streamlit rerun on unchanged values."""
    if st.query_params.get("id") != incident_id:
        st.query_params["id"] = incident_id


def render():
    inject_global_styles()

    result = load_pipeline()
    selection = resolve_packet(result)

    if selection is None:
        st.info("No incidents available. Run the pipeline with a non-empty test split.")
        st.stop()

    default_id, _ = selection
    packets = result["packets"]
    selected_id = st.selectbox(
        "Select incident",
        list(packets.keys()),
        index=list(packets.keys()).index(default_id),
    )
    sync_query_param(selected_id)
    packet = packets[selected_id]

    briefing = briefing_from_packet(packet)
    window = event_window_series(result, packet)
    eval_metrics = result.get("evaluation_metrics") or {}

    # Incident banner (native elements — no custom HTML)
    title_col, meta_col, badge_col = st.columns([3, 4, 1])
    with title_col:
        st.header(f"{packet.incident_id}")
        st.caption(f"Priority {packet.priority_label} · score {packet.priority_score:.2f}")
    with meta_col:
        st.markdown(
            f"**Window:** {packet.start_time[11:19]}Z → {packet.end_time[11:19]}Z  \n"
            f"**Duration:** {packet.duration_seconds:.0f}s · **Events:** {packet.event_count}"
        )
    with badge_col:
        st.metric("Channels", packet.channel_count)

    st.divider()

    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        # Real telemetry within the incident window
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=window['timestamps'],
            y=window['values'],
            mode='lines',
            name='Observed',
            line=dict(color='#a4e6ff', width=1.5),
        ))
        if window['event_start'] is not None and window['timestamps']:
            pos = min(window['event_start'], len(window['timestamps']) - 1)
            fig.add_vline(
                x=window['timestamps'][pos],
                line=dict(color='#ffb4ab', width=2, dash='dash'),
                annotation_text="ANOMALY ONSET",
                annotation_position="top",
                annotation=dict(font=dict(family='JetBrains Mono', size=10, color='#ffb4ab')),
            )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1c2023',
            plot_bgcolor='#0b0f11',
            font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=dict(gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
            yaxis=dict(title='Raw value', gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(family='JetBrains Mono', size=10)),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Evidence table (native)
        ev_rows = [
            {"Field": "Triggered model", "Value": f"{packet.model_name} v{packet.model_version}"},
            {"Field": "Threshold used", "Value": f"{packet.threshold_used:.3f}" if packet.threshold_used else "N/A"},
            {"Field": "Max score", "Value": f"{packet.max_anomaly_score:.3f}"},
            {"Field": "Mean score", "Value": f"{packet.mean_anomaly_score:.3f}"},
            {"Field": "Affected channels", "Value": ", ".join(packet.affected_channels)},
            {"Field": "Experiment", "Value": packet.experiment_id or "PROD-V1"},
        ]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

    with col_right:
        # Operator briefing (deterministic fallback; Granite layer pending)
        with st.container(border=True):
            st.markdown("**🤖 OPERATOR BRIEFING**")
            st.caption("Deterministic template — Granite LLM integration pending")
            st.markdown("#### 01 // Summary")
            st.markdown(briefing["summary"])
            st.markdown("#### 02 // Why flagged")
            st.markdown(briefing["why_flagged"])

        # Priority breakdown (native progress bars)
        with st.container(border=True):
            st.markdown(f"**03 // PRIORITY — {packet.priority_label} ({packet.priority_score:.2f})**")
            weights = packet.priority_weights or {}
            for key, value in sorted(
                (packet.priority_components or {}).items(),
                key=lambda kv: kv[1] * weights.get(kv[0], 0),
                reverse=True,
            ):
                weight = weights.get(key, 0)
                st.progress(
                    min(max(value, 0.0), 1.0),
                    text=f"{key.replace('_', ' ').title()} · {value * 100:.0f}% × w {weight:.2f}",
                )

        # Operator protocol
        with st.container(border=True):
            st.markdown("**04 // OPERATOR PROTOCOL**")
            for i, step in enumerate(briefing['suggestions']):
                st.checkbox(step, key=f"protocol_{i}")

        # Model & evaluation summary
        with st.container(border=True):
            st.markdown("**MODEL & EVALUATION**")
            if eval_metrics.get("f1") is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{eval_metrics['precision']:.3f}")
                m2.metric("Recall", f"{eval_metrics['recall']:.3f}")
                m3.metric("F1", f"{eval_metrics['f1']:.3f}")
            else:
                st.caption("Evaluation metrics unavailable")

        st.caption("⚠ Decision Support Only. Not for autonomous diagnosis.")


def main():
    st.set_page_config(
        page_title="MissionGuard - Incident Autopsy",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    render()


if __name__ == "__main__":
    main()
