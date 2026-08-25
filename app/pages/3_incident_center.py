"""
Incident Center - Incident list and management
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import (
    inject_global_styles, panel, badge, metric_row, data_table,
)


def load_incident_data():
    """Load incident data for the center."""
    # Mock data - in production would load from incident engine
    incidents = [
        {"incident_id": "INC-9942", "status": "new", "start_time": "14:22:01.099Z", "duration": "00:04:12", "channels": ["PRP-MAIN", "PRP-AUX"], "score": 98, "priority": "critical"},
        {"incident_id": "INC-9941", "status": "investigating", "start_time": "14:15:44.210Z", "duration": "00:10:29", "channels": ["TEL-COM"], "score": 74, "priority": "high"},
        {"incident_id": "INC-9938", "status": "reviewed", "start_time": "13:42:11.002Z", "duration": "00:02:15", "channels": ["LFS-O2"], "score": 42, "priority": "watch"},
        {"incident_id": "INC-9935", "status": "new", "start_time": "12:30:22.100Z", "duration": "00:01:45", "channels": ["CADC0872"], "score": 89, "priority": "critical"},
        {"incident_id": "INC-9932", "status": "investigating", "start_time": "11:15:33.001Z", "duration": "00:08:22", "channels": ["CADC0874", "CADC0884"], "score": 67, "priority": "high"},
        {"incident_id": "INC-9929", "status": "reviewed", "start_time": "10:05:12.000Z", "duration": "00:03:30", "channels": ["CADC0892"], "score": 35, "priority": "watch"},
        {"incident_id": "INC-9925", "status": "reviewed", "start_time": "09:45:00.000Z", "duration": "00:12:00", "channels": ["CADC0888", "CADC0890"], "score": 58, "priority": "high"},
        {"incident_id": "INC-9921", "status": "reviewed", "start_time": "08:30:00.000Z", "duration": "00:05:00", "channels": ["CADC0873"], "score": 28, "priority": "watch"},
    ]
    return incidents


def render():
    inject_global_styles()
    
    incidents = load_incident_data()
    
    # Header
    st.markdown("""
    <div class="mg-header">
        <div class="mg-header-title">MissionGuard</div>
        <div class="mg-header-nav">
            <a class="mg-header-nav-item">Overview</a>
            <a class="mg-header-nav-item">Explorer</a>
            <a class="mg-header-nav-item active">Incidents</a>
            <a class="mg-header-nav-item">Autopsy</a>
            <a class="mg-header-nav-item">Models</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Page header
    st.markdown("""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant); display: flex; justify-content: space-between; align-items: end;">
        <div>
            <h1 style="margin: 0;">Incident Center</h1>
            <div class="body-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">Active and historical incidents</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px;">
            <div class="label-caps" style="margin-bottom: 12px;">Filters</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status filter
        st.markdown("<div class='label-caps' style='margin-bottom: 8px;'>Status</div>", unsafe_allow_html=True)
        status_new = st.checkbox("New", value=True)
        status_investigating = st.checkbox("Investigating", value=True)
        status_reviewed = st.checkbox("Reviewed", value=False)
        
        # Time range
        st.markdown("<div class='label-caps' style='margin-top: 16px; margin-bottom: 8px;'>Time Range</div>", unsafe_allow_html=True)
        time_range = st.selectbox("", ["Last 1 Hour", "Last 24 Hours", "Last 7 Days", "Custom Range..."], label_visibility="collapsed")
        
        # Channels
        st.markdown("<div class='label-caps' style='margin-top: 16px; margin-bottom: 8px;'>Channels</div>", unsafe_allow_html=True)
        ch_telemetry = st.checkbox("Telemetry (TEL)", value=True)
        ch_propulsion = st.checkbox("Propulsion (PRP)", value=True)
        ch_life = st.checkbox("Life Support (LFS)", value=False)
        
        st.markdown("""
        <div style="margin-top: 24px;">
            <button class="mg-btn mg-btn-ghost" style="width: 100%;">Apply Filters</button>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    incidents_data = load_incident_data()
    
    # Filter incidents based on sidebar
    status_filter = []
    if status_new: status_filter.append("new")
    if status_investigating: status_filter.append("investigating")
    if status_reviewed: status_filter.append("reviewed")
    
    filtered = [inc for inc in incidents if inc["status"] in status_filter]
    
    # Metrics row
    metrics = [
        {"title": "Active Anomalies", "value": str(len([i for i in filtered if i["status"] != "reviewed"])), "trend": "12", "variant": "error", "icon": "warning"},
        {"title": "Avg Detection Time", "value": "1.2s", "trend": "1.2s", "variant": "primary", "icon": "speed"},
        {"title": "System Health", "value": "98.4%", "trend": "98.4%", "variant": "primary", "icon": "health_and_safety"},
    ]
    metric_row(metrics, columns=3)
    
    # Incident table
    st.markdown("""
    <div class="mg-panel" style="margin-top: 16px;">
        <div class="mg-panel-header">
            <div class="mg-panel-title">
                <span class="label-caps">Incidents</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="data-mono-md" style="color: var(--color-on-surface-variant);">Showing 1-20 of 165</span>
                <button class="mg-btn mg-btn-ghost mg-btn-sm"><span class="material-symbols-outlined">chevron_left</span></button>
                <button class="mg-btn mg-btn-ghost mg-btn-sm"><span class="material-symbols-outlined">chevron_right</span></button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Table header and rows (fixed rendering issue)
    table_html = """
    <div class="mg-table-container">
        <table class="mg-table">
            <thead>
                <tr>
                    <th class="label-caps" style="width: 80px;">Status</th>
                    <th class="label-caps" style="width: 120px;">Incident ID</th>
                    <th class="label-caps" style="width: 140px; text-align: right;">Start Time</th>
                    <th class="label-caps" style="width: 100px; text-align: right;">Duration</th>
                    <th class="label-caps">Channels</th>
                    <th class="label-caps" style="width: 80px; text-align: right;">Score</th>
                    <th class="label-caps" style="width: 50px; text-align: center;">Act</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for inc in filtered:
        status_class = "new" if inc["status"] == "new" else "investigating" if inc["status"] == "investigating" else "reviewed"
        status_colors = {
            "new": ("error", "var(--color-error)"),
            "investigating": ("warning", "var(--color-tertiary)"),
            "reviewed": ("nominal", "var(--color-on-surface-variant)"),
        }
        variant, _ = status_colors.get(inc["status"], ("nominal", "var(--color-on-surface-variant)"))
        
        table_html += f"""
        <tr style="cursor: pointer;">
            <td>
                <span class="mg-badge {variant}">
                    <span class="dot" style="background: var(--color-{'error' if variant=='error' else 'tertiary' if variant=='warning' else 'on-surface-variant'});{' animation: mg-pulse 1.5s infinite;' if variant=='error' else ''}"></span>
                    {inc["status"].capitalize()}
                </span>
            </td>
            <td class="data-mono-md" style="color: {'var(--color-error)' if variant=='error' else 'var(--color-tertiary)' if variant=='warning' else 'var(--color-on-surface-variant)'};">{inc['incident_id']}</td>
            <td class="data-mono-md" style="text-align: right;">{inc['start_time']}</td>
            <td class="data-mono-md" style="text-align: right;">{inc['duration']}</td>
            <td>{' '.join([f'<span class="mg-badge nominal">{ch}</span>' for ch in inc["channels"]])}</td>
            <td class="data-mono-md" style="text-align: right; color: {'var(--color-error)' if inc['score'] > 90 else 'var(--color-tertiary)' if inc['score'] > 70 else 'var(--color-on-surface-variant)'}; font-weight: {'bold' if inc['score'] > 70 else 'normal'};">{inc['score']}</td>
            <td style="text-align: center;"><span class="material-symbols-outlined" style="color: var(--color-primary); cursor: pointer;">open_in_new</span></td>
        </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Pagination
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; border-top: 1px solid var(--color-outline-variant);">
        <span class="label-caps" style="color: var(--color-on-surface-variant);">Showing 1-20 of 165</span>
        <div class="mg-flex mg-gap-xs">
            <button class="mg-btn mg-btn-ghost mg-btn-sm" disabled><span class="material-symbols-outlined">chevron_left</span></button>
            <button class="mg-btn mg-btn-ghost mg-btn-sm"><span class="material-symbols-outlined">chevron_right</span></button>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="MissionGuard - Incident Center",
        page_icon="⚠️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    render()


if __name__ == "__main__":
    main()