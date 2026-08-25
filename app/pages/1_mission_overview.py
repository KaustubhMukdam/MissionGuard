"""
Mission Overview - Main dashboard page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import (
    inject_global_styles, panel, kpi_card, badge, metric_row,
    chart_container, badge as status_badge, disclaimer, page_nav,
)

from app.data_bridge import build_dashboard_view, run_pipeline


@st.cache_resource(show_spinner="Running MissionGuard detection pipeline...")
def load_dashboard_data():
    """Run the production detection pipeline once and shape it for this page."""
    return build_dashboard_view(run_pipeline())


def render():
    inject_global_styles()

    # Load data — single call (Bug 6 fix: removed duplicate)
    try:
        data = load_dashboard_data()
    except FileNotFoundError as exc:
        st.error(
            "Production model artifacts not found. Run the Phase 3b pipeline to "
            f"generate models/isolation_forest_prod_v1.joblib first.\n\n{exc}"
        )
        st.stop()

    # Navigation bar (Bug 4 fix: correct page paths with .py extensions)
    page_nav([
        {"path": "app/streamlit_app.py", "label": "Overview", "icon": "rocket_launch"},
        {"path": "app/pages/2_telemetry_explorer.py", "label": "Explorer", "icon": "travel_explore"},
        {"path": "app/pages/3_incident_center.py", "label": "Incidents", "icon": "warning"},
        {"path": "app/pages/4_incident_autopsy.py", "label": "Autopsy", "icon": "science"},
        {"path": "app/pages/5_model_evaluation.py", "label": "Models", "icon": "analytics"},
    ])

    # Page header
    st.markdown(f"""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant);">
        <div style="display: flex; justify-content: space-between; align-items: end; gap: 16px;">
            <div>
                <h1 style="margin: 0;">{data['mission_name']}</h1>
                <div style="display: flex; gap: 16px; margin-top: 8px; font-family: var(--font-mono); font-size: 13px; color: var(--color-on-surface-variant);">
                    <span>Run ID: {data['run_id']}</span>
                    <span style="border-left: 1px solid var(--color-outline-variant); padding-left: 16px;">Model: {data['model_version']}</span>
                </div>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <div class="mg-flex mg-items-center mg-gap-sm" style="background: rgba(255, 213, 156, 0.15); border: 1px solid var(--color-tertiary); color: var(--color-tertiary); padding: 4px 12px; border-radius: 9999px; font-family: var(--font-mono); font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: var(--color-tertiary); animation: mg-pulse 1.5s infinite;"></span>
                    {data['health_status']}
                </div>
                <div class="mg-flex mg-items-center mg-gap-sm" style="background: rgba(255, 180, 171, 0.1); border: 1px solid var(--color-error); color: var(--color-error); padding: 4px 12px; border-radius: 9999px; font-family: var(--font-mono); font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: var(--color-error); animation: mg-pulse 1.5s infinite;"></span>
                    {data['active_incidents']} ACTIVE INCIDENTS
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    st.markdown('<div class="mg-flex mg-gap-md" style="padding: 16px 24px;">', unsafe_allow_html=True)

    kpis = data['kpis']
    cols = st.columns(4)
    for i, (key, kpi) in enumerate(kpis.items()):
        with cols[i]:
            st.markdown(kpi_card(
                title=key.replace('_', ' ').title(),
                value=kpi['value'],
                trend=kpi['trend'],
                trend_color=kpi['variant'],
                icon=kpi['icon'],
                variant=kpi['variant'],
                sparkline_data=kpi.get('sparkline'),
            ), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    col_left, col_right = st.columns([8, 4], gap="medium")

    with col_left:
        # Trend Chart (real per-segment telemetry with anomaly markers)
        trend_data = data['trend_chart']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['timestamps'],
            y=trend_data['values'],
            mode='lines',
            name=trend_data.get('channel') or 'Telemetry',
            line=dict(color='#a4e6ff', width=1.2),
        ))
        if trend_data['anomaly_positions']:
            pos = trend_data['anomaly_positions']
            fig.add_trace(go.Scatter(
                x=[trend_data['timestamps'][p] for p in pos],
                y=[trend_data['values'][p] for p in pos],
                mode='markers',
                name='Anomaly',
                marker=dict(color='#ffb4ab', size=6, symbol='x'),
            ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0b0f11',
            plot_bgcolor='#0b0f11',
            font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=dict(
                gridcolor='#3c494e',
                tickfont=dict(family='JetBrains Mono', size=9),
                showgrid=True,
            ),
            yaxis=dict(
                title='Mean value / segment',
                gridcolor='#3c494e',
                tickfont=dict(family='JetBrains Mono', size=9),
                side='left',
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(family='JetBrains Mono', size=10),
            ),
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Anomaly Log
        st.markdown("""
        <div class="mg-panel" style="margin-top: 16px;">
            <div class="mg-panel-header">
                <div class="mg-panel-title">
                    <span class="label-caps">Anomaly Log</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bug 5 fix: read from data['anomaly_log'] instead of st.session_state
        for entry in data.get('anomaly_log', []):
            severity_colors = {
                'error': ('error', 'var(--color-error)'),
                'warning': ('warning', 'var(--color-tertiary)'),
                'primary': ('primary', 'var(--color-primary)'),
                'nominal': ('nominal', 'var(--color-on-surface-variant)'),
            }
            variant, color = severity_colors.get(entry['severity'], ('nominal', 'var(--color-on-surface-variant)'))

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px; padding: 8px; border-bottom: 1px solid var(--color-outline-variant); align-items: start;">
                <div class="mg-badge {entry['severity']}" style="white-space: nowrap;">{entry['time']}</div>
                <div>
                    <div style="font-weight: 600; color: var(--color-on-surface);">{entry['event']}</div>
                    <div style="font-family: var(--font-mono); font-size: 12px; color: var(--color-on-surface-variant); margin-top: 2px;">{entry['detail']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # Active Incidents
        st.markdown("""
        <div class="mg-panel">
            <div class="mg-panel-header">
                <div class="mg-panel-title">
                    <span class="label-caps">Active Incidents</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bug 5 fix: read from data['incidents'] instead of st.session_state
        for inc in data.get('incidents', []):
            priority = inc['priority']
            badge_var = 'critical' if priority == 'critical' else 'high' if priority == 'high' else 'watch' if priority == 'watch' else 'nominal'

            st.markdown(f"""
            <div style="border: 1px solid var(--color-outline-variant); border-radius: 4px; padding: 12px; margin-bottom: 8px; background: var(--color-surface-container);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="label-caps" style="color: var(--color-on-surface);">{inc['incident_id']}</span>
                    <span class="mg-badge {badge_var}">{priority.upper()}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-family: var(--font-mono); font-size: 11px; color: var(--color-on-surface-variant);">
                    <div><span style="color: var(--color-on-surface);">Start</span><br>{inc['start_time']}</div>
                    <div><span style="color: var(--color-on-surface);">Duration</span><br>{inc['duration']}</div>
                    <div><span style="color: var(--color-on-surface);">Channels</span><br>{', '.join(inc['channels'])}</div>
                    <div><span style="color: var(--color-on-surface);">Score</span><br>{inc['score']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; padding: 16px;">
            <button class="mg-btn mg-btn-ghost" style="width: 100%;">VIEW ALL INCIDENTS</button>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="MissionGuard - Mission Overview",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render()


if __name__ == "__main__":
    main()
