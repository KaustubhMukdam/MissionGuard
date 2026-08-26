"""
Incident Center - Live incident list from the detection pipeline
"""

import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.missionguard.ui.components import inject_global_styles, metric_row
from app.data_bridge import run_pipeline, incidents_table_rows


@st.cache_resource(show_spinner="Running MissionGuard detection pipeline...")
def load_pipeline():
    return run_pipeline()


def render():
    inject_global_styles()

    result = load_pipeline()
    incidents = result["incidents"]
    rows = incidents_table_rows(result)
    metrics_eval = result.get("evaluation_metrics") or {}

    # Header
    st.markdown("""
    <div style="padding: 16px 24px; border-bottom: 1px solid var(--color-outline-variant);">
        <h1 style="margin: 0;">Incident Center</h1>
        <div class="body-md" style="color: var(--color-on-surface-variant); margin-top: 4px;">
            Incidents aggregated from the OPSSAT-AD test split (priority-ranked)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar filters (live — no apply button needed)
    with st.sidebar:
        st.markdown("<div class='label-caps' style='margin: 16px 0 8px;'>Filters</div>", unsafe_allow_html=True)
        min_score = st.slider("Minimum priority score", 0, 100, 0)
        channels = sorted({ch for inc in incidents for ch in inc.affected_channels})
        selected_channels = st.multiselect("Channels", channels, default=channels)

    filtered_rows = [
        r for r in rows
        if r["Score"] >= min_score
        and all(ch in selected_channels for ch in r["Channels"].split(", "))
    ]

    # Metrics row (real values)
    f1 = metrics_eval.get("f1")
    top_score = rows[0]["Score"] if rows else 0
    metric_row([
        {"title": "Active Incidents", "value": str(len(filtered_rows)), "trend": f"{len(rows)} total", "variant": "error" if rows else "nominal", "icon": "warning"},
        {"title": "Top Priority Score", "value": str(top_score), "trend": "ranked #1", "variant": "warning", "icon": "speed"},
        {"title": "Model F1", "value": f"{f1:.3f}" if f1 is not None else "N/A", "trend": result["model_info"]["name"] or "", "variant": "primary", "icon": "analytics"},
    ], columns=3)

    if not filtered_rows:
        st.info("No incidents match the current filters.")
        st.stop()

    # Native table (sortable, hoverable — replaces fragile hand-built HTML table)
    df = pd.DataFrame(filtered_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%d",
            ),
        },
    )

    # Deep-link into Autopsy
    st.markdown("<div class='label-caps' style='margin-top: 16px;'>Open Incident Autopsy</div>", unsafe_allow_html=True)
    selected_id = st.selectbox(
        "Incident",
        [r["Incident ID"] for r in filtered_rows],
        label_visibility="collapsed",
    )
    if st.query_params.get("id") != selected_id:
        st.query_params["id"] = selected_id
    st.markdown(
        f"<a class='mg-btn mg-btn-primary' href='/autopsy?id={selected_id}' "
        f"style='text-decoration: none;'>OPEN {selected_id} IN AUTOPSY</a>",
        unsafe_allow_html=True,
    )


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
