"""
MissionGuard - Path setup for Streamlit pages
Add this to the top of every page file to ensure src module is importable.
"""

import sys
from pathlib import Path

# Add project root to path for src module resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))