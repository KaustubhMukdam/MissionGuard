"""
Model & Evaluation - Model performance and experiment tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from app.pages._path_setup import PROJECT_ROOT
from src.missionguard.ui.components import (
    inject_global_styles, panel, chart_container, badge, metric_row,
)


def load_evaluation_data():
    """Load model evaluation data."""
    return {
        "experiment_id": "EXP-772-B",
        "model_name": "DeepPulse-S5",
        "model_status": "Active",
        "metrics": {
            "precision": 0.982,
            "recall": 0.915,
            "f1": 0.947,
        },
        "false_alarm_data": {
            "fpr": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "tpr": [0, 0.45, 0.65, 0.78, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 1.0],
        },
        "detection_delay": {
            "bins": ["0", "50", "100", "150", "200", "250", "300", "350", "400", "450", "500+"],
            "counts": [5, 12, 28, 52, 89, 145, 210, 180, 95, 42, 18],
        },
        "dataset": {
            "source": "2023-Solar-Max-Data-Split",
            "size": "45.2 TB",
            "events": 1048576,
            "start": "2023-11-04T00:00Z",
            "end": "2023-11-18T23:59Z",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "comparison": {
            "Statistical Baseline": {"precision": 0.73, "recall": 0.32, "f1": 0.44, "roc_auc": 0.89},
            "Isolation Forest (18 feat)": {"precision": 0.30, "recall": 0.81, "f1": 0.44, "roc_auc": 0.64},
            "Isolation Forest (peak)": {"precision": 0.78, "recall": 0.57, "f1": 0.66, "roc_auc": 0.63},
            "DeepPulse-S5": {"precision": 0.98, "recall": 0.92, "f1": 0.95, "roc_auc": 0.99},
        },
    }


def render():
    inject_global_styles()
    
    data = load_evaluation_data()
    
    # Header
    st.markdown("""
    <div class="mg-header">
        <div class="mg-header-title">MissionGuard</div>
        <div class="mg-header-nav">
            <a class="mg-header-nav-item">Overview</a>
            <a class="mg-header-nav-item">Explorer</a>
            <a class="mg-header-nav-item">Incidents</a>
            <a class="mg-header-nav-item">Autopsy</a>
            <a class="mg-header-nav-item active">Models</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant); display: flex; justify-content: space-between; align-items: end; gap: 16px;">
        <div>
            <h1 style="margin: 0;">Model & Evaluation</h1>
            <div class="data-mono-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">Experiment ID: <span class="primary">{data['experiment_id']}</span></div>
        </div>
        <div class="status-badge" style="background: rgba(0, 209, 255, 0.1); border: 1px solid var(--color-primary); color: var(--color-primary); padding: 4px 12px; border-radius: 9999px; font-family: var(--font-mono); font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; display: flex; align-items: center; gap: 6px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: var(--color-primary); animation: mg-pulse 1.5s infinite;"></span>
            {data['model_name']} Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    m = data['metrics']
    metric_row([
        {"title": "Precision", "value": f"{m['precision']:.1%}", "trend": "0.982", "variant": "primary", "icon": "target"},
        {"title": "Recall", "value": f"{m['recall']:.1%}", "trend": "0.915", "variant": "warning", "icon": "search"},
        {"title": "F1 Score", "value": f"{m['f1']:.3f}", "trend": "0.947", "variant": "primary", "icon": "calculate"},
    ])
    
    # Main content
    col_left, col_right = st.columns([8, 4], gap="medium")
    
    with st.container():
        col_left, col_right = st.columns([8, 4], gap="medium")
        
        with col_left:
            # False Alarm Behavior (ROC-like)
            fig = go.Figure()
            
            fpr = data['false_alarm_data']['fpr']
            tpr = data['false_alarm_data']['tpr']
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines+markers',
                name='DeepPulse-S5',
                line=dict(color='#a4e6ff', width=2),
                marker=dict(size=4, color='#a4e6ff'),
            ))
            
            # Random baseline
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='#859399', width=1, dash='dash'),
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1c2023',
                plot_bgcolor='#0b0f11',
                font=dict(family='JetBrains Mono', size=10, color='#e0e3e6'),
                margin=dict(l=60, r=20, t=10, b=40),
                xaxis=dict(
                    title='False Positive Rate',
                    gridcolor='#3c494e',
                    tickfont=dict(family='JetBrains Mono', size=9),
                ),
                yaxis=dict(
                    title='True Positive Rate',
                    gridcolor='#3c494e',
                    tickfont=dict(family='JetBrains Mono', size=9),
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    font=dict(family='JetBrains Mono', size=10),
                ),
                height=280,
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Detection Delay
            dd = data['detection_delay']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=dd['bins'],
                y=dd['counts'],
                marker_color='#a4e6ff',
                opacity=0.7,
                hovertemplate='Delay: %{x}ms<br>Count: %{y}<extra></extra>',
            ))
            # Highlight peak
            fig2.add_trace(go.Bar(
                x=[dd['bins'][5]],
                y=[dd['counts'][5]],
                marker_color='#ffddb1',
                opacity=1.0,
                showlegend=False,
            ))
            
            fig2.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1c2023',
                plot_bgcolor='#0b0f11',
                font=dict(family='JetBrains Mono', size=9, color='#e0e3e6'),
                margin=dict(l=60, r=20, t=10, b=40),
                xaxis=dict(
                    title='Detection Delay (ms)',
                    gridcolor='#3c494e',
                    tickfont=dict(family='JetBrains Mono', size=9),
                ),
                yaxis=dict(
                    gridcolor='#3c494e',
                    tickfont=dict(family='JetBrains Mono', size=9),
                ),
                showlegend=False,
                height=200,
            )
            
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        
        with col_right:
            # Model metadata
            st.markdown("""
            <div class="mg-panel">
                <div class="mg-panel-header">
                    <div class="mg-panel-title">
                        <span class="material-symbols-outlined" style="font-size: 18px;">dataset</span>
                        <span class="label-caps" style="letter-spacing: 0.1em;">Evaluation Dataset</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            ds = load_evaluation_data()['dataset']
            st.markdown(f"""
            <div class="mg-panel" style="margin-top: 16px;">
                <div class="mg-panel-header">
                    <span class="label-caps">Source Reference</span>
                </div>
                <div class="mg-panel-inlay" style="font-family: var(--font-mono); font-size: 12px; color: var(--color-primary); border-left: 2px solid var(--color-primary); padding: 8px 12px;">
                    {ds['source']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Dataset table
            st.markdown("""
            <table class="mg-table" style="margin-top: 16px;">
                <tbody>
                    <tr><td class="mg-body-md" style="color: var(--color-on-surface-variant);">Size</td><td class="mg-data-mono-md" style="text-align: right;">45.2 TB</td></tr>
                    <tr><td class="mg-body-md" style="color: var(--color-on-surface-variant);">Events</td><td class="mg-data-mono-md" style="text-align: right;">1,048,576</td></tr>
                    <tr><td class="mg-body-md" style="color: var(--color-on-surface-variant);">Timestamp Start</td><td class="mg-data-mono-md" style="text-align: right;">2023-11-04T00:00Z</td></tr>
                    <tr><td class="mg-body-md" style="color: var(--color-on-surface-variant);">Timestamp End</td><td class="mg-data-mono-md" style="text-align: right;">2023-11-18T23:59Z</td></tr>
                    <tr><td class="mg-body-md" style="color: var(--color-on-surface-variant);">Hash (SHA-256)</td><td class="mg-data-mono-md" style="text-align: right; opacity: 0.7; max-width: 120px; overflow: hidden; text-overflow: ellipsis;">e3b0c442...</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            
            st.button("EXPORT LOG REPORT", use_container_width=True, type="secondary")
        
        # Model Comparison Table
        st.markdown("""
        <div class="mg-panel" style="margin-top: 16px;">
            <div class="mg-panel-header">
                <div class="mg-panel-title">
                    <span class="label-caps">Model Comparison</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        comp = load_evaluation_data()['comparison']
        
        st.markdown("""
        <div class="mg-table-container">
            <table class="mg-table">
                <thead>
                    <tr>
                        <th class="label-caps">Model</th>
                        <th class="label-caps numeric">Precision</th>
                        <th class="label-caps numeric">Recall</th>
                        <th class="label-caps numeric">F1</th>
                        <th class="label-caps numeric">ROC-AUC</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)
        
        for name, m in comp.items():
            is_best = name == "DeepPulse-S5"
            row_style = 'style="background: rgba(0, 209, 255, 0.05);"' if is_best else ''
            best_badge = '<span class="mg-badge primary" style="font-size: 9px;">OPTIMAL</span>' if is_best else ''
            
            st.markdown(f"""
            <tr {row_style}>
                <td class="data-mono-md" style="font-weight: {'bold' if is_best else 'normal'};"{'' if not is_best else ' style="color: var(--color-primary);"'}>{name} {best_badge}</td>
                <td class="data-mono-md numeric">{m['precision']:.2f}</td>
                <td class="data-mono-md numeric">{m['recall']:.2f}</td>
                <td class="data-mono-md numeric" style="font-weight: {'bold' if is_best else 'normal'}; color: {'var(--color-primary)' if is_best else 'var(--color-on-surface)'};">{m['f1']:.3f}</td>
                <td class="data-mono-md numeric">{m['roc_auc']:.2f}</td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        st.markdown("""
        <div style="text-align: center; padding: 16px;">
            <button class="mg-btn mg-btn-ghost" style="width: 200px;">
                <span class="material-symbols-outlined" style="font-size: 14px;">download</span>
                Export Log Report
            </button>
        </div>
        """, unsafe_allow_html=True)


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