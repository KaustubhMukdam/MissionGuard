"""
MissionGuard - Main Streamlit Application Entry Point
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for src module resolution
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.missionguard.ui.components import inject_global_styles


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
    
    # Navigation is handled by Streamlit's multi-page architecture
    # Pages are auto-discovered from app/pages/


if __name__ == "__main__":
    main()