"""
MissionGuard - Main Streamlit Application Entry Point

Uses st.navigation so every page has an explicit, stable URL path
(/overview, /explorer, ...) instead of relying on auto-discovery slugs.
"""

import importlib
import sys
from pathlib import Path

import streamlit as st

# Add project root to path for src/app module resolution
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.missionguard.ui.components import inject_global_styles


def _page(module_name: str, title: str, url_path: str, default: bool = False) -> st.Page:
    """Wrap a page module's render() into an st.Page with a fixed URL path."""
    module = importlib.import_module(f"app.pages.{module_name}")
    return st.Page(module.render, title=title, url_path=url_path, default=default)


def main():
    st.set_page_config(
        page_title="MissionGuard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_global_styles()

    # Custom sidebar width
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 280px;
    }
    </style>
    """, unsafe_allow_html=True)

    nav = st.navigation([
        _page("1_mission_overview", "Overview", "overview", default=True),
        _page("2_telemetry_explorer", "Explorer", "explorer"),
        _page("3_incident_center", "Incidents", "incidents"),
        _page("4_incident_autopsy", "Autopsy", "autopsy"),
        _page("5_model_evaluation", "Models", "models"),
    ])

    nav.run()


if __name__ == "__main__":
    main()
