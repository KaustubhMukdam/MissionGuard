"""
Telemetry Explorer - Interactive telemetry visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from app.pages._path_setup import PROJECT_ROOT
from src.missionguard.ui.components import (
    inject_global_styles, panel, chart_container, badge,
)


def load_telemetry_data():
    """Load telemetry data for explorer."""
    # Mock data - in production would load from data pipeline
    channels = {
        "CADC0872": {"name": "X-Band Transmitter Temp", "unit": "°C", "status": "error"},
        "CADC0892": {"name": "Star Tracker 1 Alignment", "unit": "arcsec", "status": "nominal"},
        "CADC0874": {"name": "Reaction Wheel A RPM", "unit": "RPM", "status": "warning"},
        "CADC0884": {"name": "Battery Bank C Charge", "unit": "%", "status": "nominal"},
        "CADC0873": {"name": "Solar Array Current", "unit": "A", "status": "nominal"},
    }
    
    # Generate time series data
    base_time = datetime.now() - timedelta(hours=1)
    timestamps = [base_time + timedelta(seconds=i) for i in range(3600)]
    
    # Generate realistic telemetry for each channel
    data = {}
    for ch_id, ch_info in channels.items():
        np.random.seed(hash(ch_id) % 2**32)
        base = 20 if "Temp" in ch_info["name"] else 100 if "RPM" in ch_info["name"] else 80 if "Charge" in ch_info["name"] else 5
        noise = np.random.normal(0, 0.5, 3600)
        values = base + noise
        if ch_id == "CADC0872":  # Add anomaly spike
            values[1800:2000] += np.linspace(0, 15, 200)
        data[ch_id] = values
    
    return channels, timestamps, data


def render():
    inject_global_styles()
    
    channels, timestamps, ts_data = load_telemetry_data()
    
    # Page header
    st.markdown("""
    <div class="mg-header">
        <div class="mg-header-title">MissionGuard</div>
        <div class="mg-header-nav">
            <a class="mg-header-nav-item">Overview</a>
            <a class="mg-header-nav-item active">Explorer</a>
            <a class="mg-header-nav-item">Incidents</a>
            <a class="mg-header-nav-item">Autopsy</a>
            <a class="mg-header-nav-item">Models</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant); display: flex; justify-content: space-between; align-items: end;">
        <div>
            <h1 style="margin: 0;">Telemetry Explorer</h1>
            <div class="body-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">SUBSYSTEM: COMM-LINK | ID: CL-992-ALPHA</div>
        </div>
        <div style="display: flex; gap: 8px;">
            <button class="mg-btn mg-btn-ghost">EXPORT CSV</button>
            <button class="mg-btn mg-btn-primary">CREATE MODEL</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout: Sidebar (channels) | Main (chart) | Right panel (inspection)
    col_sidebar, col_main, col_inspect = st.columns([1.5, 5, 1.5], gap="medium")
    
    # Initialize session state for selected channel
    if 'selected_channel' not in st.session_state:
        st.session_state.selected_channel = list(ts_data.keys())[0]
    if 'time_range' not in st.session_state:
        st.session_state.time_range = (0, 3600)
    
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px;">
            <div class="label-caps" style="margin-bottom: 12px;">Active Channels</div>
            <input type="text" placeholder="Filter channels..." style="width: 100%; background: var(--color-surface-container); border: 1px solid var(--color-outline-variant); border-radius: 4px; padding: 8px; font-family: var(--font-mono); font-size: 12px; color: var(--color-on-surface); margin-bottom: 12px;">
        </div>
        """, unsafe_allow_html=True)
        
        for ch_id, ch_info in load_telemetry_data()[0].items():
            is_active = ch_id == st.session_state.selected_channel
            status_colors = {
                'nominal': ('nominal', 'var(--color-primary)'),
                'warning': ('warning', 'var(--color-tertiary)'),
                'error': ('error', 'var(--color-error)'),
            }
            variant, dot_color = status_colors.get(ch_info["status"], ('nominal', 'var(--color-primary)'))
            
            active_class = 'active' if is_active else ''
            st.markdown(f"""
            <div class="mg-nav-item {active_class}" style="cursor: pointer; margin-bottom: 4px;" onclick="window.parent.postMessage({{channel: '{ch_id}'}}, '*')">
                <div style="flex: 1;">
                    <div class="data-mono-md" style="color: var(--color-on-surface);">{ch_info['name']}</div>
                    <div class="label-caps" style="color: var(--color-on-surface-variant);">{ch_id}</div>
                </div>
                <div class="w-2 h-2 rounded-full" style="background: {dot_color};"></div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.session_state:
        pass
    
    # Main chart area
    ch_id = st.session_state.selected_channel
    ch_info = load_telemetry_data()[0][ch_id]
    values = load_telemetry_data()[2][ch_id]
    timestamps_list = load_telemetry_data()[1]
    
    # Create time range slider
    st.markdown("""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant); display: flex; justify-content: space-between; align-items: center;">
        <div class="mg-flex mg-items-center mg-gap-md">
            <span class="material-symbols-outlined" style="color: var(--color-primary);">travel_explore</span>
            <h1 class="mg-headline-md" style="margin: 0;">Telemetry Explorer</h1>
        </div>
        <div class="mg-flex mg-items-center mg-gap-sm">
            <button class="mg-btn mg-btn-ghost mg-btn-sm">EXPORT CSV</button>
            <button class="mg-btn mg-btn-primary mg-btn-sm">CREATE MODEL</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Time range selector
    time_start, time_end = st.slider(
        "Time Range",
        0, len(load_telemetry_data()[1]) - 1,
        st.session_state.time_range,
        format="%d sec",
        label_visibility="collapsed"
    )
    st.session_state.time_range = (time_start, time_end)
    
    # Slice data
    t_slice = timestamps[time_start:time_end]
    v_slice = values[time_start:time_end]
    
    # Create Plotly chart
    fig = go.Figure()
    
    # Observed line
    fig.add_trace(go.Scatter(
        x=t_slice,
        y=v_slice,
        mode='lines',
        name='Observed',
        line=dict(color='#a4e6ff', width=1.5),
    ))
    
    # Expected line (simple moving average as proxy)
    expected = pd.Series(values).rolling(window=50, min_periods=1).mean().values[time_start:time_end]
    fig.add_trace(go.Scatter(
        x=t_slice,
        y=expected,
        mode='lines',
        name='Expected (Model)',
        line=dict(color='#bbc9cf', width=1.5, dash='dash'),
    ))
    
    # Anomaly regions
    anomaly_indices = [i for i, v in enumerate(values) if i >= time_start and i < time_end and v > 25]
    if anomaly_indices:
        fig.add_vrect(
            x0=t_slice[0], x1=t_slice[-1],
            fillcolor="#ffb4ab", opacity=0.15,
            layer="below", line_width=0,
        )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#101416',
        plot_bgcolor='#0b0f11',
        font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
        margin=dict(l=60, r=60, t=10, b=40),
        xaxis=dict(gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
        yaxis=dict(gridcolor='#3c494e', tickfont=dict(family='JetBrains Mono', size=9)),
        yaxis2=dict(
            title='Anomaly Score',
            overlaying='y',
            side='right',
            showgrid=False,
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
    
    # Time range slider
    st.markdown("""
    <div style="margin-top: 16px; padding: 12px; background: var(--color-surface-container); border: 1px solid var(--color-outline-variant); border-radius: 4px;">
        <div class="mg-flex mg-items-center mg-justify-between mg-mb-sm">
            <span class="label-caps">Time Range Selection</span>
            <span class="data-mono-md" style="color: var(--color-primary);">T- 24H — NOW</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Inspection panel
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px; background: var(--color-surface-container); border: 1px solid var(--color-outline-variant); border-radius: 8px; margin-top: 24px;">
            <div class="label-caps" style="margin-bottom: 16px;">Inspection Details</div>
        </div>
        """, unsafe_allow_html=True)
        
        ch_info = load_telemetry_data()[0][st.session_state.selected_channel]
        current_val = values[-1]
        
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div class="label-caps" style="margin-bottom: 8px; color: var(--color-on-surface-variant);">Current Value</div>
            <div class="mg-data-mono-lg" style="color: var(--color-primary);">{current_val:.1f}<span class="mg-headline-md" style="color: var(--color-on-surface-variant); margin-left: 8px;">{load_telemetry_data()[0][load_telemetry_data()[2].keys().__iter__().__next__()]['unit'] if False else '°C'}</span></div>
            <div class="data-mono-md" style="color: var(--color-error); margin-top: 4px; display: flex; align-items: center; gap: 4px;">
                <span class="material-symbols-outlined" style="font-size: 14px;">arrow_upward</span>
                +4.3°C Δ Expected
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Anomaly assessment
        st.markdown("""
        <div style="background: var(--color-error-container); border: 1px solid var(--color-error); border-radius: 4px; padding: 12px; margin-bottom: 16px;">
            <div class="mg-flex mg-items-center mg-gap-sm mg-mb-sm">
                <span class="material-symbols-outlined" style="color: var(--color-on-error); font-size: 20px;">warning</span>
                <span class="label-caps" style="color: var(--color-on-error); font-weight: 700;">CRITICAL ANOMALY</span>
            </div>
            <div class="data-mono-md" style="color: var(--color-on-error); opacity: 0.9;">
                Score: 0.89 (High Confidence)<br>Pattern: Thermal Runaway
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metadata
        st.markdown("""
        <div style="border-top: 1px solid var(--color-outline-variant); padding-top: 16px;">
            <div class="mg-flex mg-justify-between mg-items-center mg-py-xs" style="border-bottom: 1px solid var(--color-outline-variant);">
                <span class="label-caps" style="color: var(--color-on-surface-variant);">Sensor ID</span>
                <span class="data-mono-md" style="text-align: right;">TX-THERM-A</span>
            </div>
            <div class="mg-flex mg-justify-between mg-items-center mg-py-xs" style="border-bottom: 1px solid var(--color-outline-variant);">
                <span class="label-caps" style="color: var(--color-on-surface-variant);">Sample Rate</span>
                <span class="data-mono-md" style="text-align: right;">10 Hz</span>
            </div>
            <div class="mg-flex mg-justify-between mg-items-center mg-py-xs">
                <span class="label-caps" style="color: var(--color-on-surface-variant);">Last Calibrated</span>
                <span class="data-mono-md" style="text-align: right;">T- 48:00:00</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("VIEW RAW LOGS", use_container_width=True, type="secondary")


def main():
    st.set_page_config(
        page_title="MissionGuard - Telemetry Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom sidebar width
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 280px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    render()


if __name__ == "__main__":
    main()