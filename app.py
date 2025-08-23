import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import base64
from datetime import datetime, timedelta
import os

# Import utility modules
from utils.data_processor import DataProcessor
from utils.ai_insights import AIInsights
from utils.trend_detector import TrendDetector
from utils.visualizations import Visualizations
from utils.export_utils import ExportUtils

# Page configuration
st.set_page_config(
    page_title="Creative Analytics AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'trends' not in st.session_state:
    st.session_state.trends = None

# Initialize utility classes
@st.cache_resource
def get_utilities():
    return {
        'data_processor': DataProcessor(),
        'ai_insights': AIInsights(),
        'trend_detector': TrendDetector(),
        'visualizations': Visualizations(),
        'export_utils': ExportUtils()
    }

utils = get_utilities()

# Main title and description
st.title("🎨 Creative Analytics AI")
st.markdown("**AI-powered data analytics tool for creatives** - Upload your data and discover insights with advanced AI")

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing your data..."):
                st.session_state.data = utils['data_processor'].process_file(uploaded_file)
            st.success(f"✅ Data loaded successfully! ({len(st.session_state.data)} rows)")
            
            # Data overview
            st.subheader("📊 Data Overview")
            st.write(f"**Rows:** {len(st.session_state.data)}")
            st.write(f"**Columns:** {len(st.session_state.data.columns)}")
            
            # Show column types
            col_types = st.session_state.data.dtypes.value_counts()
            st.write("**Column Types:**")
            for dtype, count in col_types.items():
                st.write(f"- {dtype}: {count} columns")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.session_state.data = None

    st.divider()
    
    # AI Settings
    st.header("🤖 AI Settings")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success("✅ OpenAI API key configured")
        
        if st.session_state.data is not None:
            if st.button("🔍 Generate AI Insights", type="primary"):
                with st.spinner("Analyzing your data with AI..."):
                    try:
                        st.session_state.insights = utils['ai_insights'].generate_insights(st.session_state.data)
                        st.session_state.trends = utils['trend_detector'].detect_trends(st.session_state.data)
                        st.success("✅ AI analysis complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating insights: {str(e)}")
    else:
        st.warning("⚠️ OpenAI API key not found in environment variables")
        st.info("Set OPENAI_API_KEY environment variable to enable AI features")

# Main content area
if st.session_state.data is not None:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 AI Insights", 
        "📈 Trends & Patterns", 
        "📉 Visualizations", 
        "📋 Export"
    ])
    
    with tab1:
        st.header("Data Overview & Quality Assessment")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Data Sample")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            st.subheader("Data Statistics")
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
            
        with col2:
            st.subheader("Data Quality")
            quality_report = utils['data_processor'].assess_data_quality(st.session_state.data)
            
            # Missing values
            st.metric("Missing Values", f"{quality_report['missing_percentage']:.1f}%")
            
            # Duplicate rows
            st.metric("Duplicate Rows", quality_report['duplicates'])
            
            # Data types
            st.subheader("Column Information")
            col_info = utils['data_processor'].get_column_info(st.session_state.data)
            st.dataframe(col_info, use_container_width=True)
            
            # Missing values heatmap
            if quality_report['missing_percentage'] > 0:
                st.subheader("Missing Values Pattern")
                missing_fig = utils['visualizations'].create_missing_values_heatmap(st.session_state.data)
                st.plotly_chart(missing_fig, use_container_width=True)
    
    with tab2:
        st.header("🤖 AI-Powered Insights")
        
        if st.session_state.insights:
            # Display insights
            insights_data = st.session_state.insights
            
            # Key findings
            if 'key_findings' in insights_data:
                st.subheader("🔍 Key Findings")
                for i, finding in enumerate(insights_data['key_findings'], 1):
                    st.write(f"**{i}.** {finding}")
            
            # Recommendations
            if 'recommendations' in insights_data:
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(insights_data['recommendations'], 1):
                    st.write(f"**{i}.** {rec}")
            
            # Data insights
            if 'data_insights' in insights_data:
                st.subheader("📊 Data Insights")
                for insight in insights_data['data_insights']:
                    with st.expander(insight.get('title', 'Insight')):
                        st.write(insight.get('description', ''))
                        if 'metric' in insight:
                            st.metric(insight['metric']['name'], insight['metric']['value'])
            
            # Creative metrics focus
            if 'creative_metrics' in insights_data:
                st.subheader("🎨 Creative Performance Metrics")
                metrics = insights_data['creative_metrics']
                
                cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with cols[i]:
                        st.metric(metric_name.replace('_', ' ').title(), metric_value)
        else:
            st.info("Click 'Generate AI Insights' in the sidebar to analyze your data with AI")
    
    with tab3:
        st.header("📈 Trends & Pattern Detection")
        
        if st.session_state.trends:
            trends_data = st.session_state.trends
            
            # Trend summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Detected Trends")
                if 'trends' in trends_data:
                    for trend in trends_data['trends']:
                        with st.expander(f"📈 {trend.get('column', 'Trend')} - {trend.get('trend_type', 'Unknown')}"):
                            st.write(f"**Trend:** {trend.get('trend_type', 'Unknown')}")
                            st.write(f"**Strength:** {trend.get('strength', 'Unknown')}")
                            st.write(f"**Description:** {trend.get('description', 'No description available')}")
                            
                            if 'correlation' in trend:
                                st.write(f"**Correlation:** {trend['correlation']:.3f}")
            
            with col2:
                st.subheader("🔍 Pattern Analysis")
                if 'patterns' in trends_data:
                    for pattern in trends_data['patterns']:
                        with st.expander(f"🔍 {pattern.get('type', 'Pattern')}"):
                            st.write(pattern.get('description', 'No description available'))
                            if 'confidence' in pattern:
                                st.progress(pattern['confidence'])
                                st.write(f"Confidence: {pattern['confidence']:.1%}")
            
            # Time series analysis if applicable
            time_cols = utils['data_processor'].identify_time_columns(st.session_state.data)
            if time_cols:
                st.subheader("📅 Time Series Analysis")
                selected_time_col = st.selectbox("Select time column:", time_cols)
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected_metric = st.selectbox("Select metric to analyze:", numeric_cols)
                    
                    if st.button("Generate Time Series Chart"):
                        fig = utils['visualizations'].create_time_series_chart(
                            st.session_state.data, selected_time_col, selected_metric
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate AI insights first to see trend analysis")
    
    with tab4:
        st.header("📉 Interactive Visualizations")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select visualization type:",
            ["Correlation Heatmap", "Distribution Analysis", "Scatter Plot", "Bar Chart", "Box Plot", "Histogram"]
        )
        
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if chart_type == "Correlation Heatmap" and len(numeric_cols) > 1:
            fig = utils['visualizations'].create_correlation_heatmap(st.session_state.data[numeric_cols])
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Distribution Analysis" and numeric_cols:
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            fig = utils['visualizations'].create_distribution_plot(st.session_state.data, selected_col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Select Y-axis:", [col for col in numeric_cols if col != x_col])
            
            color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
            color_col = color_col if color_col != "None" else None
            
            fig = utils['visualizations'].create_scatter_plot(st.session_state.data, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Bar Chart" and categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Select category:", categorical_cols)
            with col2:
                num_col = st.selectbox("Select value:", numeric_cols)
            
            fig = utils['visualizations'].create_bar_chart(st.session_state.data, cat_col, num_col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot" and categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Select category:", categorical_cols)
            with col2:
                num_col = st.selectbox("Select value:", numeric_cols)
            
            fig = utils['visualizations'].create_box_plot(st.session_state.data, cat_col, num_col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Histogram" and numeric_cols:
            selected_col = st.selectbox("Select column for histogram:", numeric_cols)
            bins = st.slider("Number of bins:", 10, 100, 30)
            fig = utils['visualizations'].create_histogram(st.session_state.data, selected_col, bins)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please ensure your data has the required column types for the selected visualization.")
    
    with tab5:
        st.header("📋 Export Data & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Export Data")
            
            # Data export options
            export_format = st.selectbox("Select export format:", ["CSV", "Excel", "JSON"])
            
            if st.button("Download Processed Data"):
                if export_format == "CSV":
                    csv_data = utils['export_utils'].to_csv(st.session_state.data)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel_data = utils['export_utils'].to_excel(st.session_state.data)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "JSON":
                    json_data = utils['export_utils'].to_json(st.session_state.data)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            st.subheader("📝 Export Insights Report")
            
            if st.session_state.insights or st.session_state.trends:
                report_format = st.selectbox("Report format:", ["PDF Report", "JSON Insights", "Text Summary"])
                
                if st.button("Generate Report"):
                    if report_format == "JSON Insights":
                        insights_json = utils['export_utils'].export_insights_json(
                            st.session_state.insights, st.session_state.trends
                        )
                        st.download_button(
                            label="📥 Download Insights JSON",
                            data=insights_json,
                            file_name=f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    elif report_format == "Text Summary":
                        text_report = utils['export_utils'].export_text_report(
                            st.session_state.insights, st.session_state.trends
                        )
                        st.download_button(
                            label="📥 Download Text Report",
                            data=text_report,
                            file_name=f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.info("PDF export functionality would require additional libraries")
            else:
                st.info("Generate AI insights first to export reports")

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to Creative Analytics AI! 🎨
    
    This powerful tool helps creatives analyze their data and discover actionable insights using advanced AI.
    
    ### Features:
    - **📁 Multi-format Support**: Upload CSV, Excel, or JSON files
    - **🤖 AI-Powered Insights**: Automatic analysis using GPT-4o
    - **📈 Trend Detection**: Advanced pattern recognition and trend analysis
    - **📊 Interactive Visualizations**: Dynamic charts with drill-down capabilities
    - **🔍 Data Quality Assessment**: Comprehensive data profiling
    - **📋 Export Capabilities**: Download insights and processed data
    
    ### Perfect for:
    - Social media managers analyzing engagement data
    - Content creators tracking performance metrics
    - Marketing teams evaluating campaign results
    - Designers measuring creative impact
    - Agencies reporting to clients
    
    **Get started by uploading your data file in the sidebar!**
    """)
    
    # Sample data structure guide
    with st.expander("💡 Data Structure Guidelines"):
        st.markdown("""
        For best results, structure your data with columns like:
        
        **Creative Metrics:**
        - `engagement_rate`, `click_through_rate`, `conversion_rate`
        - `likes`, `shares`, `comments`, `views`
        - `impressions`, `reach`, `frequency`
        
        **Performance Data:**
        - `date` or `timestamp` for time series analysis
        - `campaign_name`, `content_type`, `platform`
        - `cost`, `revenue`, `roi`
        
        **Audience Data:**
        - `age_group`, `gender`, `location`
        - `device_type`, `source`, `medium`
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    Creative Analytics AI - Powered by OpenAI GPT-4o | Built with Streamlit
</div>
""", unsafe_allow_html=True)
