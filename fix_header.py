with open('app/pages/1_mission_overview.py', 'r') as f:
    content = f.read()

# Find and replace the header section
old = '''# Header
    st.markdown(f"""
    <div class="mg-header">
        <div class="mg-header-title">MissionGuard</div>
        <div class="mg-header-nav">
            <a class="mg-header-nav-item active">Overview</a>
            <a class="mg-header-nav-item">Explorer</a>
            <a class="mg-header-nav-item">Incidents</a>
            <a class="mg-header-nav-item">Autopsy</a>
            <a class="mg-header-nav-item">Models</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Page header'''

new = '''# Load data
    data = load_dashboard_data()
    
    # Navigation bar
    page_nav([
        {"path": "streamlit_app.py", "label": "Overview", "icon": "rocket_launch"},
        {"path": "Telemetry_Explorer", "label": "Explorer", "icon": "travel_explore"},
        {"path": "Incident_Center", "label": "Incidents", "icon": "warning"},
        {"path": "Incident_Autopsy", "label": "Autopsy", "icon": "science"},
        {"path": "Model_Evaluation", "label": "Models", "icon": "analytics"},
    ])
    
    # Page header'''

content = content.replace(old, new)

with open('app/pages/1_mission_overview.py', 'w') as f:
    f.write(content)
print('Done')