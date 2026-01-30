# app.py - Complete Streamlit Frontend
import streamlit as st
import requests
import json
from PIL import Image
import io
import tempfile
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
import time

# Page configuration
st.set_page_config(
    page_title="EnviroAudit - Environmental Compliance Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .risk-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #DC2626;
    }
    .risk-high {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #F59E0B;
    }
    .risk-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #EAB308;
    }
    .risk-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #10B981;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# Title
st.markdown('<h1 class="main-header">🌍 EnviroAudit</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Environmental Compliance Monitoring System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/environment.png", width=80)
    st.title("Configuration")
    
    # API Settings
    st.subheader("API Settings")
    api_url = st.text_input(
        "API Server URL",
        value=st.session_state.api_url,
        help="URL of the EnviroAudit API server"
    )
    st.session_state.api_url = api_url
    
    # Analysis Method
    st.subheader("Analysis Method")
    analysis_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Image URL", "Sample Images"],
        label_visibility="collapsed"
    )
    
    # Project Information
    st.subheader("Project Information")
    project_id = st.text_input("Project ID", value="PROJ-2024-001")
    latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
    longitude = st.number_input("Longitude", value=-74.0060, format="%.6f")
    date_taken = st.date_input("Date Taken", value=datetime.now())
    
    # Quick Actions
    st.subheader("Quick Actions")
    if st.button("📊 View Dashboard", use_container_width=True):
        st.session_state.show_dashboard = True
    else:
        st.session_state.show_dashboard = False
    
    if st.button("🔄 Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.analysis_results = None
        st.session_state.image = None
        st.rerun()
    
    # Help Section
    with st.expander("ℹ️ Help & Instructions"):
        st.markdown("""
        **How to use:**
        1. Select analysis method (Upload/URL/Sample)
        2. Configure project information
        3. Click 'Analyze Image'
        4. View results and generate report
        
        **Sample Images:**
        - Construction sites
        - Mining operations
        - Natural landscapes
        - Urban development
        
        **API Status:** Check if server is running
        """)
        
        # Test API connection
        if st.button("Test API Connection"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is running")
                else:
                    st.error(f"❌ API error: {response.status_code}")
            except:
                st.error("❌ Cannot connect to API")

# Main Content Tabs
if st.session_state.get('show_dashboard', False):
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📋 Analysis", "⚙️ Settings"])
else:
    tab1, tab2 = st.tabs(["📋 Analysis", "⚙️ Settings"])

# Analysis Tab
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input Image")
        
        if analysis_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image for analysis"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.session_state.image = image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                with st.expander("Image Details"):
                    st.write(f"**Format:** {image.format}")
                    st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                    st.write(f"**Mode:** {image.mode}")
        
        elif analysis_method == "Image URL":
            image_url = st.text_input(
                "Enter image URL",
                value="https://images.unsplash.com/photo-1581094794329-c8112a89af12",
                help="URL of an image to analyze"
            )
            
            if st.button("Load Image from URL") and image_url:
                with st.spinner("Downloading image..."):
                    try:
                        response = requests.get(image_url, timeout=10)
                        image = Image.open(io.BytesIO(response.content))
                        st.session_state.image = image
                        st.image(image, caption="Image from URL", use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to load image: {str(e)}")
        
        else:  # Sample Images
            st.subheader("Sample Images")
            sample_images = {
                "Construction Site": "https://images.unsplash.com/photo-1581094794329-c8112a89af12",
                "Mining Operation": "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09",
                "Natural Landscape": "https://images.unsplash.com/photo-1501854140801-50d01698950b",
                "Urban Development": "https://images.unsplash.com/photo-1541888946425-d81bb19240f5"
            }
            
            selected_sample = st.selectbox("Choose a sample:", list(sample_images.keys()))
            
            if st.button("Load Sample Image"):
                with st.spinner("Loading sample..."):
                    try:
                        response = requests.get(sample_images[selected_sample], timeout=10)
                        image = Image.open(io.BytesIO(response.content))
                        st.session_state.image = image
                        st.image(image, caption=f"Sample: {selected_sample}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to load sample: {str(e)}")
    
    with col2:
        st.header("🔍 Analysis")
        
        # Analyze button
        analyze_disabled = st.session_state.image is None
        if st.button("🔍 Analyze Image", disabled=analyze_disabled, type="primary", use_container_width=True):
            if st.session_state.image:
                with st.spinner("Analyzing image. This may take 10-20 seconds..."):
                    try:
                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        st.session_state.image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        # Prepare request
                        files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                        data = {
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                            "project_id": project_id,
                            "date": date_taken.isoformat()
                        }
                        
                        # Call API
                        response = requests.post(
                            f"{api_url}/analyze",
                            files=files,
                            data=data,
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.analysis_results = result
                            
                            # Add to history
                            history_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "project_id": project_id,
                                "primary_label": result.get('classification', {}).get('primary_label'),
                                "risk_level": result.get('compliance', {}).get('risk_level'),
                                "confidence": result.get('classification', {}).get('primary_confidence'),
                                "analysis_id": result.get('metadata', {}).get('analysis_id')
                            }
                            st.session_state.history.append(history_entry)
                            
                            st.success("✅ Analysis complete!")
                        else:
                            st.error(f"❌ Analysis failed: {response.status_code}")
                            st.error(f"Error: {response.text[:200]}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API server. Make sure it's running.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Risk Level Banner
            risk_level = results.get('compliance', {}).get('risk_level', 'UNKNOWN')
            risk_class = f"risk-{risk_level.lower()}" if risk_level != 'UNKNOWN' else ""
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="margin: 0;">Risk Level: {risk_level}</h3>
                <p style="margin: 5px 0 0 0;">{results.get('compliance', {}).get('recommended_action', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.subheader("📊 Analysis Metrics")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Primary Classification",
                    value=results.get('classification', {}).get('primary_label', 'N/A')
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Confidence",
                    value=results.get('classification', {}).get('primary_confidence', 'N/A')
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                detection = "Yes" if results.get('classification', {}).get('is_construction_activity') else "No"
                st.metric(
                    label="Construction Detected",
                    value=detection
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed Results
            with st.expander("📝 Detailed Analysis", expanded=True):
                # Classification Details
                st.write("**Classification Details:**")
                predictions = results.get('classification', {}).get('predictions', [])
                for pred in predictions[:3]:  # Show top 3
                    st.write(f"- {pred.get('label')}: {pred.get('confidence')}")
                
                # Caption
                st.write("**Image Description:**")
                caption = results.get('caption', {}).get('basic_caption', 'No caption available')
                st.info(caption)
                
                # Compliance Indicators
                st.write("**Compliance Indicators:**")
                indicators = results.get('compliance', {}).get('indicators', {})
                for key, value in indicators.items():
                    display_key = key.replace('_', ' ').title()
                    display_value = "✅ Yes" if value is True else ("❌ No" if value is False else value)
                    st.write(f"- {display_key}: {display_value}")
            
            # Report Generation
            st.subheader("📄 Report")
            
            # Download buttons
            col_x, col_y = st.columns(2)
            
            with col_x:
                # JSON download
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"enviroaudit_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_y:
                # Text report
                report_text = results.get('report', {}).get('text', '')
                st.download_button(
                    label="📄 Download Text Report",
                    data=report_text,
                    file_name=f"enviroaudit_report_{project_id}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # View full report
            if st.button("👁️ View Full Report"):
                st.text_area("Complete Report", report_text, height=300)

# Dashboard Tab (if enabled)
if st.session_state.get('show_dashboard', False):
    with tab1:  # Dashboard tab
        st.title("📈 EnviroAudit Dashboard")
        
        # Fetch history data
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            
            # Metrics
            st.subheader("Project Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", len(df))
            
            with col2:
                high_risk = len(df[df['risk_level'].isin(['CRITICAL', 'HIGH'])])
                st.metric("High Risk Cases", high_risk)
            
            with col3:
                avg_confidence = pd.to_numeric(df['confidence'].str.strip('%'), errors='coerce').mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col4:
                unique_projects = df['project_id'].nunique()
                st.metric("Projects", unique_projects)
            
            # Charts
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Risk Distribution")
                if 'risk_level' in df.columns:
                    risk_counts = df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No risk level data available")
            
            with col_b:
                st.subheader("Recent Activity")
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['hour'] = df['timestamp'].dt.floor('H')
                    hourly_counts = df.groupby('hour').size().reset_index(name='count')
                    fig = px.line(
                        hourly_counts,
                        x='hour',
                        y='count',
                        title='Analyses per Hour'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timestamp data available")
            
            # Recent analyses table
            st.subheader("Recent Analyses")
            display_df = df.copy()
            if 'timestamp' in display_df.columns:
                display_df = display_df.sort_values('timestamp', ascending=False)
            st.dataframe(
                display_df.head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No analysis history available. Perform some analyses to see dashboard data.")

# Settings Tab
with tab2 if not st.session_state.get('show_dashboard', False) else tab3:
    st.title("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Configuration")
        
        # API health check
        if st.button("Check API Status"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success(f"✅ API is running (Version: {response.json().get('version', '1.0.0')})")
                else:
                    st.error(f"❌ API error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Cannot connect to API: {str(e)}")
        
        # API endpoints info
        with st.expander("API Endpoints"):
            st.markdown("""
            **Available Endpoints:**
            - `GET /` - API information
            - `GET /health` - Health check
            - `POST /analyze` - Upload image for analysis
            - `POST /analyze-url` - Analyze image from URL
            - `GET /analyses` - Get analysis history (if database enabled)
            
            **Example cURL:**
            ```bash
            curl -X POST "http://localhost:8000/analyze-url" \\
              -H "Content-Type: application/json" \\
              -d '{
                "image_url": "https://example.com/image.jpg",
                "project_id": "test-001"
              }'
            ```
            """)
    
    with col2:
        st.subheader("System Information")
        
        # Display system info
        st.markdown(f"""
        **Current Configuration:**
        - API URL: `{api_url}`
        - Analysis Method: `{analysis_method}`
        - Project ID: `{project_id}`
        
        **Session:**
        - Analyses in history: {len(st.session_state.history)}
        - Current image: {'Loaded' if st.session_state.image else 'None'}
        - Current results: {'Available' if st.session_state.analysis_results else 'None'}
        """)
        
        # Export session data
        if st.button("Export Session Data"):
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "api_url": api_url,
                "project_id": project_id,
                "history": st.session_state.history,
                "current_results": st.session_state.analysis_results
            }
            
            json_str = json.dumps(session_data, indent=2, default=str)
            st.download_button(
                label="Download Session Data",
                data=json_str,
                file_name=f"enviroaudit_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
# New tab for enhanced features
if not st.session_state.get('show_dashboard', False):
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Analysis", "🛰️ Satellite", "📊 Dashboard", "⚙️ Settings"])
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Analysis", "🛰️ Satellite", "📊 Dashboard", "📈 Advanced", "⚙️ Settings"])

# Satellite Analysis Tab
with tab2:
    st.title("🛰️ Satellite Imagery Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Location Configuration")
        
        # Location input
        location_method = st.radio(
            "Location input method:",
            ["Coordinates", "Map Selection"]
        )
        
        if location_method == "Coordinates":
            lat = st.number_input("Latitude", value=40.7128, format="%.6f")
            lon = st.number_input("Longitude", value=-74.0060, format="%.6f")
        else:
            # Simple map for demo
            st.info("Map selection requires additional setup")
            lat = st.number_input("Latitude", value=40.7128, format="%.6f")
            lon = st.number_input("Longitude", value=-74.0060, format="%.6f")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        bbox_size = st.slider(
            "Area Size (km)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Size of area to analyze"
        )
        
        analysis_date = st.date_input(
            "Imagery Date",
            value=datetime.now() - timedelta(days=30)
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Single Date", "Change Detection"]
        )
        
        if analysis_type == "Change Detection":
            comparison_date = st.date_input(
                "Comparison Date",
                value=datetime.now() - timedelta(days=365)
            )
        
        project_id = st.text_input("Project ID", value=f"SAT-{datetime.now().strftime('%Y%m%d')}")
        
        # Action buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🌍 Analyze Location", use_container_width=True):
                with st.spinner("Fetching satellite imagery and analyzing..."):
                    try:
                        response = requests.post(
                            f"{api_url}/analyze-location",
                            json={
                                "latitude": float(lat),
                                "longitude": float(lon),
                                "date": analysis_date.strftime("%Y-%m-%d"),
                                "bbox_size": bbox_size / 111,  # Convert km to degrees
                                "project_id": project_id
                            },
                            timeout=180
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.satellite_results = result
                            st.success("✅ Satellite analysis complete!")
                        else:
                            st.error(f"❌ Analysis failed: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col_b:
            if analysis_type == "Change Detection":
                if st.button("🔄 Compare Changes", use_container_width=True):
                    with st.spinner("Comparing changes over time..."):
                        try:
                            response = requests.post(
                                f"{api_url}/compare-location",
                                json={
                                    "latitude": float(lat),
                                    "longitude": float(lon),
                                    "date1": comparison_date.strftime("%Y-%m-%d"),
                                    "date2": analysis_date.strftime("%Y-%m-%d"),
                                    "bbox_size": bbox_size / 111,
                                    "project_id": f"CHANGE-{project_id}"
                                },
                                timeout=180
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state.change_results = result
                                st.success("✅ Change analysis complete!")
                            else:
                                st.error(f"❌ Comparison failed: {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Results Visualization")
        
        # Display satellite results
        if st.session_state.get('satellite_results'):
            results = st.session_state.satellite_results
            
            # Display satellite image
            if 'satellite_data' in results and 'satellite_image' in results['satellite_data']:
                st.image(results['satellite_data']['satellite_image'], caption="Satellite Imagery", use_column_width=True)
            
            # Display analysis results
            with st.expander("📊 Analysis Results", expanded=True):
                if 'classification' in results:
                    st.write("**Classification:**")
                    st.write(f"- Primary: {results['classification'].get('primary_label', 'N/A')}")
                    st.write(f"- Confidence: {results['classification'].get('primary_confidence', 'N/A')}")
                
                if 'object_detection' in results and results['object_detection'].get('available'):
                    st.write("**Object Detection:**")
                    stats = results['object_detection'].get('statistics', {})
                    st.write(f"- Heavy Machinery: {stats.get('heavy_machinery_count', 0)}")
                    st.write(f"- Vehicles: {stats.get('vehicle_count', 0)}")
                    st.write(f"- Total Objects: {stats.get('total_count', 0)}")
                
                if 'compliance' in results:
                    st.write("**Compliance Assessment:**")
                    compliance = results['compliance']
                    risk_color = {
                        'CRITICAL': 'red',
                        'HIGH': 'orange',
                        'MEDIUM': 'yellow',
                        'LOW': 'green'
                    }.get(compliance.get('risk_level', 'UNKNOWN'), 'gray')
                    
                    st.markdown(f"Risk Level: :{risk_color}[**{compliance.get('risk_level', 'N/A')}**]")
                    st.write(f"Action: {compliance.get('recommended_action', 'N/A')}")
            
            # Download options
            if st.button("📥 Download Satellite Report"):
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Click to Download",
                    data=json_str,
                    file_name=f"satellite_analysis_{project_id}.json",
                    mime="application/json"
                )
        
        # Display change detection results
        elif st.session_state.get('change_results'):
            results = st.session_state.change_results
            
            col_x, col_y = st.columns(2)
            
            with col_x:
                if 'comparison_images' in results and results['comparison_images'].get('image1'):
                    st.image(results['comparison_images']['image1'], caption=f"Date: {results.get('date1', 'N/A')}", use_column_width=True)
            
            with col_y:
                if 'comparison_images' in results and results['comparison_images'].get('image2'):
                    st.image(results['comparison_images']['image2'], caption=f"Date: {results.get('date2', 'N/A')}", use_column_width=True)
            
            # Change statistics
            with st.expander("🔄 Change Analysis", expanded=True):
                st.write(f"**Change Percentage:** {results.get('change_percentage', 0):.1f}%")
                st.write(f"**Change Detected:** {'✅ Yes' if results.get('change_detected') else '❌ No'}")
                
                if 'changes' in results:
                    changes = results['changes']
                    st.write("**Detailed Changes:**")
                    st.write(f"- Risk Score Change: {changes.get('risk_score_change', 0):+.1f}")
                    st.write(f"- Heavy Machinery Change: {changes.get('heavy_machinery_change', 0):+d}")
                    st.write(f"- Overall Change: {changes.get('overall_change', 'N/A')}")
        
        else:
            st.info("👈 Configure location and run analysis to see results")
            
            # Example coordinates
            st.subheader("Example Locations:")
            examples = st.columns(3)
            
            with examples[0]:
                if st.button("🏗️ New York Construction", use_container_width=True):
                    st.session_state.example_lat = 40.7128
                    st.session_state.example_lon = -74.0060
                    st.rerun()
            
            with examples[1]:
                if st.button("🏞️ Yosemite National Park", use_container_width=True):
                    st.session_state.example_lat = 37.8651
                    st.session_state.example_lon = -119.5383
                    st.rerun()
            
            with examples[2]:
                if st.button("⛏️ Chilean Copper Mine", use_container_width=True):
                    st.session_state.example_lat = -22.2758
                    st.session_state.example_lon = -68.8972
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>🌍 <b>EnviroAudit</b> - AI-powered environmental compliance monitoring system</p>
    <p>Built with FastAPI, Transformers, and Streamlit • Version 1.0.0</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for long operations
if st.session_state.get('analyzing', False):
    time.sleep(1)
    st.rerun()