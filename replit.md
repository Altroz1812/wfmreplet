# Creative Analytics AI

## Overview

Creative Analytics AI is a Streamlit-based data analytics application that provides AI-powered insights and visualizations for datasets. The application focuses on making data analysis accessible through an intuitive web interface, combining automated data processing, trend detection, AI-generated insights, and interactive visualizations. It's designed to help users understand their data through advanced analytics while maintaining simplicity in presentation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **Session Management**: Streamlit session state for persistent data storage across user interactions
- **User Interface**: Single-page application with sidebar navigation and main content area
- **Caching Strategy**: Resource caching using `@st.cache_resource` for utility class initialization

### Backend Architecture
- **Modular Design**: Utility-based architecture with separate modules for distinct functionalities
- **Core Components**:
  - `DataProcessor`: Handles file upload, data cleaning, and format conversion
  - `AIInsights`: Generates AI-powered data insights using OpenAI GPT models
  - `TrendDetector`: Performs statistical analysis, pattern recognition, and anomaly detection
  - `Visualizations`: Creates interactive charts and graphs using Plotly
  - `ExportUtils`: Manages data export in multiple formats (CSV, Excel, JSON)

### Data Processing Pipeline
- **Input Handling**: Multi-format file support (CSV, Excel, JSON) with automatic format detection
- **Data Cleaning**: Automated data preprocessing and quality assessment
- **Analysis Flow**: Sequential processing through trend detection, AI analysis, and visualization generation
- **Error Handling**: Comprehensive exception handling with user-friendly error messages

### AI Integration
- **Provider**: OpenAI API integration for generating insights
- **Model**: GPT-4o (latest model as of May 2024) for advanced analysis capabilities
- **Analysis Types**: General insights, creative-specific insights, and data summarization
- **Configuration**: Environment-based API key management

### Visualization Strategy
- **Library**: Plotly for interactive charts and graphs
- **Chart Types**: Correlation heatmaps, trend visualizations, distribution plots, and custom analytics charts
- **Styling**: Consistent color palette and layout configuration for professional appearance
- **Interactivity**: Hover effects, zoom, and selection capabilities for enhanced user experience

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing for statistical operations
- **Plotly**: Interactive visualization library for charts and graphs

### AI and Machine Learning
- **OpenAI**: AI insights generation using GPT models
- **Scikit-learn**: Machine learning algorithms for trend detection and clustering
- **SciPy**: Scientific computing for statistical analysis

### Data Processing
- **JSON**: Built-in JSON handling for data import/export
- **IO**: File handling and data streaming operations
- **Base64**: Data encoding for file operations

### Statistical Analysis
- **Linear Regression**: Trend detection and forecasting
- **Correlation Analysis**: Pattern identification in datasets
- **K-Means Clustering**: Data grouping and segmentation
- **Anomaly Detection**: Outlier identification using statistical methods

### Environment Configuration
- **OpenAI API Key**: Required environment variable for AI functionality
- **File System**: Local file operations for data processing and caching