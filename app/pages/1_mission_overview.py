"""
Mission Overview - Main dashboard page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from app.pages._path_setup import PROJECT_ROOT  # Ensures src module is importable
from src.missionguard.ui.components import (
    inject_global_styles, panel, kpi_card, badge, metric_row,
    chart_container, badge as status_badge, disclaimer, page_nav,
)


def load_dashboard_data():
    """Load data for the dashboard."""
    # In production, this would load from the incident engine
    # For now, return mock data
    return {
        "mission_name": "Orbital Sentinel-9",
        "run_id": "OS9-A-2024-Q4",
        "model_version": "Sentinel-Core v4.2.1",
        "health_status": "WATCH (AMBER)",
        "active_incidents": 3,
        "kpis": {
            "power": {"value": "98.2%", "trend": "+0.4% /hr", "variant": "primary", "icon": "bolt", "sparkline": [95, 96, 97, 98, 98, 99]},
            "thermal": {"value": "312K", "trend": "+5K warning limit", "variant": "warning", "icon": "thermostat", "sparkline": [305, 308, 310, 312, 312, 312]},
            "propulsion": {"value": "NOMINAL", "trend": "Delta-v: 450 m/s", "variant": "primary", "icon": "speed", "sparkline": [90, 92, 94, 95, 95, 96]},
            "comm_link": {"value": "72%", "trend": "Degraded bandwidth", "variant": "error", "icon": "satellite_alt", "sparkline": [95, 88, 82, 78, 75, 72]},
        },
        "trend_chart": {
            "timestamps": [datetime.now() - timedelta(minutes=i*15) for i in range(20, 0, -1)],
            "bus_voltage": [28.1, 28.2, 28.0, 28.3, 28.1, 28.4, 28.2, 28.5, 28.3, 28.6, 28.4, 28.7, 28.5, 28.8, 28.6, 28.9, 28.7, 29.0, 28.8, 29.1],
            "payload_power": [420, 420, 415, 425, 430, 428, 432, 435, 430, 428, 435, 438, 435, 432, 430, 428, 425, 430, 432, 430],
        },
        "anomaly_log": [
            {"time": "14:22:09", "event": "Comm-Link Degraded", "detail": "Packet loss > 15% on Tx-Alpha. Rerouting...", "severity": "error"},
            {"time": "13:05:44", "event": "Thermal Spk Core-2", "detail": "Temp rise +4K/min. Coolant loop active.", "severity": "warning"},
            {"time": "11:45:00", "event": "Routine Telemetry Sync", "detail": "Handshake successful. V4.2.1 nominal.", "severity": "primary"},
            {"time": "09:12:33", "event": "Payload Sequence Init", "detail": "Auto-sequencer started.", "severity": "nominal"},
        ],
        "incidents": [
            {"incident_id": "INC-9942", "start_time": "14:22:01", "duration": "00:04:12", "channels": ["PRP-MAIN", "PRP-AUX"], "score": 98, "priority": "critical"},
            {"incident_id": "INC-9941", "start_time": "14:15:44", "duration": "00:10:29", "channels": ["TEL-COM"], "score": 74, "priority": "high"},
            {"incident_id": "INC-9938", "start_time": "13:42:11", "duration": "00:02:15", "channels": ["LFS-O2"], "score": 42, "priority": "watch"},
        ],
    }


def render():
    inject_global_styles()
    
    # Load data
    data = load_dashboard_data()
    
    # Load data
    data = load_dashboard_data()
    
    # Navigation bar
    page_nav([
        {"path": "streamlit_app.py", "label": "Overview", "icon": "rocket_launch"},
        {"path": "Telemetry_Explorer", "label": "Explorer", "icon": "travel_explore"},
        {"path": "Incident_Center", "label": "Incidents", "icon": "warning"},
        {"path": "Incident_Autopsy", "label": "Autopsy", "icon": "science"},
        {"path": "Model_Evaluation", "label": "Models", "icon": "analytics"},
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
        # Trend Chart
        trend_data = data['trend_chart']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['timestamps'],
            y=trend_data['bus_voltage'],
            mode='lines',
            name='Bus (V)',
            line=dict(color='#a4e6ff', width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=trend_data['timestamps'],
            y=trend_data['payload_power'],
            mode='lines',
            name='Payload (W)',
            line=dict(color='#ffddb1', width=1, dash='dash'),
            yaxis='y2'
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
                title='Bus (V)',
                gridcolor='#3c494e',
                tickfont=dict(family='JetBrains Mono', size=9),
                side='left',
            ),
            yaxis2=dict(
                title='Payload (W)',
                gridcolor='#3c494e',
                tickfont=dict(family='JetBrains Mono', size=9),
                overlaying='y',
                side='right',
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
        
        for entry in st.session_state.get('anomaly_log', []):
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
        
        incidents_data = st.session_state.get('incidents', [])
        for inc in incidents_data:
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