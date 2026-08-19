"""
MissionGuard UI Components Library
Reusable HTML/CSS components for the MissionGuard Streamlit dashboard.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def inject_custom_css():
    """Inject custom CSS into Streamlit app."""
    with open("app/style.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render page header with title and optional subtitle."""
    st.markdown(f"""
    <div class="mg-page-header">
        <h1>{icon} {title}</h1>
        {f'<p class="body-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def panel(
    title: str,
    content: Union[str, None] = None,
    icon: str = "",
    children: Optional[List] = None,
    header_actions: Optional[str] = None,
) -> None:
    """Render a MissionGuard panel with header and content."""
    header = f"""
    <div class="mg-panel-header">
        <div class="mg-panel-title">
            {'<span class="mg-panel-icon material-symbols-outlined">' + icon + '</span>' if icon else ''}
            <span class="label-caps">{title}</span>
        </div>
        {header_actions or ''}
    </div>
    """
    
    content_html = content or ""
    if children:
        content_html = "".join(children)
    
    st.markdown(f"""
    <div class="mg-panel">
        {header}
        <div>{content_html}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(
    title: str,
    value: str,
    trend: str = "",
    trend_color: str = "primary",
    icon: str = "",
    variant: str = "primary",
    sparkline_data: Optional[List[float]] = None,
) -> str:
    """Generate KPI card HTML."""
    trend_class = f"mg-kpi-trend {trend_color}"
    icon_html = f'<span class="mg-kpi-icon material-symbols-outlined">{icon}</span>' if icon else ""
    
    sparkline_html = ""
    if sparkline_data:
        max_val = max(sparkline_data) if sparkline_data else 1
        bars = "".join([
            f'<div class="mg-sparkline-bar {variant}" style="height: {max(4, int(v/max_val*28))}px;"></div>'
            for v in sparkline_data
        ])
        sparkline_html = f'<div class="mg-sparkline-bars">{bars}</div>'
    
    return f"""
    <div class="mg-kpi-card {variant}">
        <div class="mg-kpi-header">
            <div class="mg-kpi-title">
                <span class="mg-kpi-icon material-symbols-outlined">{icon}</span>
                <span class="label-caps">{title}</span>
            </div>
        </div>
        <div class="mg-kpi-value" style="color: var(--color-{variant});">{value}</div>
        <div class="mg-kpi-trend {trend_color}">{trend}</div>
        {sparkline_html}
    </div>
    """


def badge(label: str, variant: str = "nominal", pulse: bool = False, count: Optional[int] = None) -> str:
    """Generate status badge HTML."""
    pulse_class = "pulse" if pulse else ""
    count_html = f'<span style="margin-left: 4px; background: var(--color-{variant}-container); padding: 0 4px; border-radius: 9999px;">{count}</span>' if count is not None else ""
    return f'<span class="mg-badge {variant} {pulse_class}"><span class="dot"></span>{label}{count_html}</span>'


def status_badge(text: str, variant: str = "nominal", count: Optional[int] = None) -> str:
    """Alias for badge for status display."""
    return badge(text, variant, count=count)


def data_table(
    df: pd.DataFrame,
    column_config: Optional[Dict] = None,
    key: Optional[str] = None,
    height: Optional[int] = None,
) -> Any:
    """Render a styled data table."""
    # Apply custom CSS classes to dataframe
    return st.dataframe(
        df,
        column_config=column_config,
        key=key,
        height=height,
        use_container_width=True,
        hide_index=True,
    )


def metric_row(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """Render a row of KPI cards."""
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i]:
            st.markdown(kpi_card(**metric), unsafe_allow_html=True)


def incident_row(
    incident: Dict[str, Any],
    on_click: Optional[callable] = None,
) -> str:
    """Generate incident table row HTML."""
    priority = incident.get("priority_score", 0)
    if priority >= 0.75:
        badge_variant = "critical"
    elif priority >= 0.5:
        badge_variant = "high"
    elif priority >= 0.25:
        badge_variant = "watch"
    else:
        badge_variant = "nominal"
    
    return f"""
    <tr style="cursor: pointer;">
        <td class="mono">{incident.get('incident_id', 'INC-0000')}</td>
        <td class="mono">{incident.get('start_time', '')}</td>
        <td class="numeric mono">{incident.get('duration_seconds', 0):.0f}s</td>
        <td>{', '.join(incident.get('affected_channels', []))}</td>
        <td class="numeric mono">{incident.get('max_anomaly_score', 0):.3f}</td>
        <td>{badge(incident.get('priority_label', 'NOMINAL').lower(), variant='critical' if priority >= 0.75 else 'high' if priority >= 0.5 else 'watch' if priority >= 0.25 else 'nominal')}</td>
        <td class="text-center">
            <span class="material-symbols-outlined" style="color: var(--color-primary); cursor: pointer;">open_in_new</span>
        </td>
    </tr>
    """


def evidence_section(
    title: str,
    content: str,
    icon: str = "",
    variant: str = "",
) -> str:
    """Generate evidence section HTML for Incident Autopsy."""
    icon_html = f'<span class="material-symbols-outlined" style="color: var(--color-error);">{icon}</span>' if icon else ""
    variant_class = f" {variant}" if variant else ""
    
    return f"""
    <div class="mg-evidence-section{variant_class}">
        <div class="mg-evidence-header">
            {f'<span class="material-symbols-outlined" style="color: var(--color-error);">{icon}</span>' if icon else ''}
            <span class="label-caps">{title}</span>
        </div>
        <div class="mg-evidence-content">{content}</div>
    </div>
    """


def logic_vector_table(data: Dict[str, str]) -> str:
    """Generate logic vector table HTML."""
    rows = ""
    for i, (label, value) in enumerate(data.items()):
        border = "" if i == 0 else 'border-top: 1px solid var(--color-outline-variant);'
        rows += f"""
        <div class="mg-logic-vector-row" style="{border}">
            <span class="mg-logic-vector-label">{label}</span>
            <span class="mg-logic-vector-value">{value}</span>
        </div>
        """
    return f'<div class="mg-logic-vector">{rows}</div>'


def priority_breakdown(score: float, components: Dict[str, float], weights: Dict[str, float]) -> str:
    """Generate priority breakdown HTML."""
    items = ""
    for key, value in components.items():
        weight = weights.get(key, 0)
        color_map = {
            "max_anomaly_score": "error",
            "mean_anomaly_score": "tertiary",
            "duration_factor": "primary",
            "channel_count_factor": "tertiary",
            "event_count_factor": "primary",
            "recurrence_factor": "secondary",
        }
        color = color_map.get(key, "secondary")
        items += f"""
        <div class="mg-priority-breakdown-item">
            <div class="mg-priority-breakdown-color" style="background: var(--color-{color});"></div>
            <span>{key.replace('_', ' ').title()}</span>
            <span style="margin-left: auto; color: var(--color-on-surface-variant);">{value*100:.0f}%</span>
        </div>
        """
    
    return f"""
    <div class="mg-flex mg-items-center mg-gap-md">
        <div class="mg-priority-donut" style="--error-pct: {components.get('max_anomaly_score',0)*360}deg; --tertiary-pct: {(components.get('max_anomaly_score',0)+components.get('duration_factor',0))*360}deg;">
            <div class="mg-priority-center">{int(score*100)}</div>
        </div>
        <div class="mg-priority-breakdown">{items}</div>
    </div>
    """


def checklist(items: List[Dict[str, str]], key_prefix: str = "checklist") -> List[bool]:
    """Render a checklist and return checked states."""
    checked = []
    for i, item in enumerate(items):
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            checked_state = st.checkbox("", key=f"{key_prefix}_{i}", label_visibility="collapsed")
            checked.append(checked_state)
        with col2:
            st.markdown(f'<span class="body-md">{item.get("text", "")}</span>', unsafe_allow_html=True)
    return checked


def disclaimer() -> str:
    """Generate disclaimer footer HTML."""
    return """
    <div class="mg-disclaimer">
        <p class="mg-disclaimer-text">
            ⚠ Decision Support Only. Not for autonomous diagnosis.
        </p>
    </div>
    """


def chart_container(title: str, legend_items: List[Dict], content: str) -> str:
    """Wrap chart in container with header and legend."""
    legend_html = ""
    if legend_items:
        legend_items_html = ""
        for item in legend_items:
            legend_items_html += f"""
            <div class="mg-chart-legend-item">
                <div class="mg-chart-legend-color" style="background: {item['color']};"></div>
                <span class="label-caps">{item['label']}</span>
            </div>
            """
        legend_html = f'<div class="mg-chart-legend">{legend_items_html}</div>'
    
    return f"""
    <div class="mg-chart-container">
        <div class="mg-chart-header">
            <span class="label-caps">{title}</span>
            {legend_html}
        </div>
        <div>{content}</div>
    </div>
    """


def header(title: str, actions: Optional[str] = None) -> str:
    """Generate header HTML."""
    return f"""
    <div class="mg-header">
        <div class="mg-header-title">{title}</div>
        <div class="mg-header-nav">{actions or ''}</div>
    </div>
    """


def sidebar_nav(items: List[Dict], active_key: str) -> str:
    """Generate sidebar navigation HTML."""
    items_html = ""
    for item in items:
        active_class = " active" if item["key"] == active_key else ""
        items_html += f"""
        <a href="#" class="mg-nav-item{active_class}" data-key="{item['key']}">
            <span class="mg-nav-icon material-symbols-outlined">{item.get('icon', '')}</span>
            <span>{item['label']}</span>
        </a>
        """
    return f'<nav class="mg-sidebar">{items_html}</nav>'


def page_nav(pages: List[Dict]) -> None:
    """Render Streamlit page navigation links."""
    for page in pages:
        st.page_link(
            page=page["path"],
            label=page["label"],
            icon=page.get("icon", ""),
            use_container_width=False,
        )


def inject_global_styles() -> None:
    """Inject all custom CSS and fonts."""
    # Load Google Fonts
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&icon_names=rocket_launch,ads_click,ac_unit,speed,settings_input_antenna,list_alt,terminal,warning,smart_toy,bolt,thermostat,speed,satellite_alt,travel_explore,open_in_new,chevron_left,chevron_right,filter_list,search,settings,notifications&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    # Load custom CSS
    with open("app/style.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ============================================================
# Evidence Packet Functions (for Incident Autopsy)
# ============================================================

def serialize_evidence_packet(packet: "EvidencePacket", format: str = "json") -> str:
    """Serialize evidence packet to string."""
    if format == "json":
        return packet.to_json()
    elif format == "yaml":
        try:
            import yaml
            return yaml.dump(packet.to_dict(), default_flow_style=False)
        except ImportError:
            raise ValueError("PyYAML required for YAML format")
    else:
        raise ValueError(f"Unknown format: {format}")


def build_llm_prompt(evidence_packet: "EvidencePacket") -> str:
    """Build grounded prompt for LLM briefing."""
    evidence_json = evidence_packet.to_json(indent=2)
    return f"""You are a spacecraft operations analyst. Generate a concise operator briefing for the incident below.

INCIDENT EVIDENCE:
{evidence_json}

CONSTRAINTS:
1. Use ONLY the evidence provided. Do NOT invent telemetry values, channels, or timestamps.
2. Do NOT claim causal root cause or physical diagnosis.
3. Distinguish observations from hypotheses.
4. State uncertainty explicitly.
5. Provide 2-3 actionable investigation suggestions.
6. End with: "⚠ Decision Support Only. Not for autonomous diagnosis."

STRUCTURE YOUR RESPONSE AS:
**Summary** (1-2 sentences): What happened?
**Why Flagged** (1-2 sentences): Which evidence triggered the alert?
**Investigation Suggestions** (2-3 bullets): What should the operator check?
"""


def validate_evidence_packet(packet: "EvidencePacket") -> List[str]:
    """
    Validate evidence packet for completeness.
    
    Returns:
        List of validation warnings (empty if valid)
    """
    import math
    warnings = []
    
    if not packet.incident_id:
        warnings.append("Missing incident_id")
    if not packet.anomaly_events:
        warnings.append("No anomaly events in packet")
    if packet.max_anomaly_score is None:
        warnings.append("Missing max_anomaly_score")
    if packet.priority_score is None:
        warnings.append("Missing priority_score")
    if packet.model_name is None:
        warnings.append("Missing model_name")
    if packet.evaluation_f1 is None:
        warnings.append("Missing evaluation metrics (f1, precision, recall)")
    
    # Check for NaN values
    for key, value in packet.to_dict().items():
        if isinstance(value, float) and math.isnan(value):
            warnings.append(f"NaN value in field: {key}")
    
    return warnings