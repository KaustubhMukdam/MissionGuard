"""
Incident Autopsy - Deep dive investigation view
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import (
    inject_global_styles, panel, evidence_section, logic_vector_table,
    priority_breakdown, checklist, disclaimer, chart_container, badge,
)


def load_incident_data(incident_id: str = "INC-4022"):
    """Load incident data for autopsy."""
    # Mock data - in production would load from incident engine
    return {
        "incident_id": incident_id,
        "title": "Thermal Runaway",
        "timestamp": "2024-11-20T08:42:15Z",
        "subsystem": "Cryo-Loop",
        "severity": "CRITICAL",
        "priority_score": 94,
        "priority_components": {
            "max_anomaly_score": 0.65,
            "duration_factor": 0.25,
            "channel_count_factor": 0.10,
            "mean_anomaly_score": 0.15,
            "event_count_factor": 0.10,
            "recurrence_factor": 0.05,
        },
        "priority_weights": {
            "max_anomaly_score": 0.35,
            "duration_factor": 0.20,
            "channel_count_factor": 0.15,
            "mean_anomaly_score": 0.15,
            "event_count_factor": 0.10,
            "recurrence_factor": 0.05,
        },
        "evidence": {
            "triggered_rule": "THR-EXC-T12",
            "observed_value": "1140 K",
            "nominal_limit": "850 K",
            "channel": "T-12",
            "duration_seconds": 252,
            "max_anomaly_score": 0.89,
            "pattern": "Thermal Runaway",
        },
        "model": {"name": "IsolationForestDetector", "version": "1.0.0", "exp_id": "EXP-005"},
        "evaluation": {"precision": 0.98, "recall": 0.91, "f1": 0.94},
        "time_series": {
            "timestamps": [datetime.now() - timedelta(minutes=i*5) for i in range(60, 0, -1)],
            "baseline": [300 + 20*np.sin(i/10) for i in range(60)],
            "anomaly": [300 + 20*np.sin(i/10) + (0 if i < 40 else (i-40)*25) for i in range(60)],
        },
        "operator_protocol": [
            "Isolate coolant sector 4 and reroute flow via backup line B.",
            "Verify physical integrity of Valve V-89 via optical sensors.",
            "Initiate manual bleed procedure for Channel T-12.",
        ],
    }


def render():
    inject_global_styles()

    # Get incident ID from query params or default
    incident_id = st.query_params.get("id", "INC-4022")
    data = load_incident_data(incident_id)

    # Header
    st.markdown("""
    <div class="mg-header">
        <div class="mg-header-title">MissionGuard</div>
        <div class="mg-header-nav">
            <a class="mg-header-nav-item">Overview</a>
            <a class="mg-header-nav-item">Explorer</a>
            <a class="mg-header-nav-item active">Incidents</a>
            <a class="mg-header-nav-item active">Autopsy</a>
            <a class="mg-header-nav-item">Models</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Incident header banner
    st.markdown(f"""
    <div class="mg-panel" style="margin: 16px 24px;">
        <div class="mg-panel-header">
            <div class="mg-flex mg-items-center mg-gap-sm">
                <span class="material-symbols-outlined" style="color: var(--color-error); font-size: 28px;">warning</span>
                <div>
                    <h1 style="margin: 0;">{data['incident_id']}: {data['title']}</h1>
                    <div class="data-mono-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">Timestamp: {data['timestamp']} | Subsystem: {data['subsystem']}</div>
                </div>
            </div>
            <div class="mg-flex mg-items-center mg-gap-sm">
                <span class="mg-badge critical">CRITICAL</span>
                <span class="mg-badge nominal" style="background: var(--color-surface-container-high); border-color: var(--color-outline-variant); color: var(--color-on-surface-variant);">T-12</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bug 9 fix: single st.columns() definition, both columns used once each
    col_left, col_right = st.columns([2, 1], gap="medium")

    with col_left:
        # Primary Telemetry Chart
        ts_data = data['time_series']

        fig = go.Figure()

        # Baseline
        fig.add_trace(go.Scatter(
            x=ts_data['timestamps'],
            y=ts_data['baseline'],
            mode='lines',
            name='Baseline Temp',
            line=dict(color='#a4e6ff', width=1.5),
        ))

        # Anomaly channel
        fig.add_trace(go.Scatter(
            x=ts_data['timestamps'],
            y=ts_data['anomaly'],
            mode='lines',
            name='Channel T-12 Temp',
            line=dict(color='#ffb4ab', width=1.5),
        ))

        # Failure point marker
        fig.add_vline(
            x=ts_data['timestamps'][40],
            line=dict(color='#ffb4ab', width=2, dash='dash'),
            annotation_text="FAILURE POINT T+04:12",
            annotation_position="top",
            annotation=dict(font=dict(family='JetBrains Mono', size=10, color='#ffb4ab'))
        )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1c2023',
            plot_bgcolor='#0b0f11',
            font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=dict(gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
            yaxis=dict(
                title='Temperature (K)',
                gridcolor='#3c494e',
                tickfont=dict(family='JetBrains Mono', size=9),
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='right', x=1,
                font=dict(family='JetBrains Mono', size=10),
            ),
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Time range buttons
        st.markdown("""
        <div class="mg-flex mg-gap-xs" style="margin-top: 8px;">
            <button class="mg-btn mg-btn-ghost mg-btn-sm">1M</button>
            <button class="mg-btn mg-btn-primary mg-btn-sm">5M</button>
            <button class="mg-btn mg-btn-ghost mg-btn-sm">15M</button>
        </div>
        """, unsafe_allow_html=True)

        # Evidence sections (Bug 9 fix: moved into col_left, was incorrectly after container)
        st.markdown(evidence_section(
            "EVIDENCE",
            logic_vector_table(data['evidence']),
            icon="list_alt"
        ), unsafe_allow_html=True)

        st.markdown(evidence_section(
            "AFFECTED CHANNELS",
            "<div class='mg-flex mg-flex-col mg-gap-sm'><span class='mg-badge critical'>T-12</span><span class='mg-badge nominal'>Coolant Loop B</span></div>",
            icon="router"
        ), unsafe_allow_html=True)

    with col_right:
        # AI Operator Briefing
        st.markdown("""
        <div class="mg-panel">
            <div class="mg-panel-header">
                <div class="mg-panel-title">
                    <span class="material-symbols-outlined" style="color: var(--color-primary);">smart_toy</span>
                    <span class="label-caps" style="color: var(--color-primary); letter-spacing: 0.1em;">AI OPERATOR BRIEFING</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(evidence_section(
            "01 // SUMMARY",
            f"At T+04:12, thermal sensors on Channel T-12 recorded an anomalous spike, exceeding the safe operational threshold of 850 K. The rate of temperature increase (dT/dt) bypassed safety margins, indicating a failure in the secondary coolant loop regulation valve.",
            icon="description"
        ), unsafe_allow_html=True)

        st.markdown(evidence_section(
            "02 // LOGIC VECTOR",
            logic_vector_table(data['evidence']),
            icon="psychology"
        ), unsafe_allow_html=True)

        st.markdown(priority_breakdown(
            data['priority_score'] / 100,
            data['priority_components'],
            data['priority_weights']
        ), unsafe_allow_html=True)

        # Operator Protocol
        st.markdown(evidence_section(
            "04 // OPERATOR PROTOCOL",
            "",
            icon="checklist"
        ), unsafe_allow_html=True)

        for i, step in enumerate(data['operator_protocol']):
            st.checkbox(step, key=f"protocol_{i}")

        # Model info
        st.markdown(evidence_section(
            "MODEL INFO",
            f"""<div class="mg-flex mg-flex-col mg-gap-sm">
                <div><span class="label-caps" style="color: var(--color-on-surface-variant);">Model</span><span class="data-mono-md">{data['model']['name']}</span></div>
                <div><span class="label-caps" style="color: var(--color-on-surface-variant);">Version</span><span class="data-mono-md">{data['model']['version']}</span></div>
                <div><span class="label-caps" style="color: var(--color-on-surface-variant);">Experiment</span><span class="data-mono-md">{data['model']['exp_id']}</span></div>
            </div>""",
            icon="settings"
        ), unsafe_allow_html=True)

        # Evaluation
        st.markdown(evidence_section(
            "EVALUATION",
            f"""<div class="mg-flex mg-flex-col mg-gap-sm">
                <div class="mg-flex mg-justify-between"><span class="label-caps" style="color: var(--color-on-surface-variant);">Precision</span><span class="data-mono-md">{data['evaluation']['precision']:.0%}</span></div>
                <div class="mg-flex mg-justify-between"><span class="label-caps" style="color: var(--color-on-surface-variant);">Recall</span><span class="data-mono-md">{data['evaluation']['recall']:.0%}</span></div>
                <div class="mg-flex mg-justify-between"><span class="label-caps" style="color: var(--color-on-surface-variant);">F1</span><span class="data-mono-md" style="color: var(--color-primary); font-weight: 700;">{data['evaluation']['f1']:.0%}</span></div>
            </div>""",
            icon="analytics"
        ), unsafe_allow_html=True)

        st.markdown(disclaimer(), unsafe_allow_html=True)


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
