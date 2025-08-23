import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from openai import OpenAI

class AIInsights:
    """Generate AI-powered insights using OpenAI."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
    
    def generate_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive AI insights from the DataFrame."""
        if not self.client:
            raise Exception("OpenAI API key not configured")
        
        # Prepare data summary for AI analysis
        data_summary = self._prepare_data_summary(df)
        
        try:
            # Generate general insights
            insights = self._generate_general_insights(data_summary, df)
            
            # Generate creative-specific insights if applicable
            creative_insights = self._generate_creative_insights(data_summary, df)
            
            # Combine insights
            combined_insights = {
                **insights,
                **creative_insights,
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_overview': data_summary
            }
            
            return combined_insights
            
        except Exception as e:
            raise Exception(f"Error generating AI insights: {str(e)}")
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare a comprehensive data summary for AI analysis."""
        # Basic statistics
        summary = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {}
            for col in categorical_cols[:10]:  # Limit to top 10 to avoid token limits
                value_counts = df[col].value_counts().head(10)
                summary['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': value_counts.to_dict()
                }
        
        # Time series detection
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        if len(datetime_cols) > 0:
            summary['time_series_columns'] = list(datetime_cols)
            for col in datetime_cols:
                summary[f'{col}_range'] = {
                    'min': str(df[col].min()),
                    'max': str(df[col].max()),
                    'span_days': (df[col].max() - df[col].min()).days
                }
        
        # Identify potential creative metrics
        creative_keywords = ['engagement', 'click', 'conversion', 'view', 'impression', 'reach', 'like', 'share', 'comment', 'revenue', 'cost', 'roi']
        potential_creative_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in creative_keywords):
                potential_creative_cols.append(col)
        
        summary['potential_creative_metrics'] = potential_creative_cols
        
        return summary
    
    def _generate_general_insights(self, data_summary: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate general data insights using AI."""
        
        prompt = f"""
        As an expert data analyst, analyze this dataset and provide actionable insights.
        
        Dataset Summary:
        - Shape: {data_summary['shape']['rows']} rows × {data_summary['shape']['columns']} columns
        - Columns: {', '.join(data_summary['columns'][:20])}{'...' if len(data_summary['columns']) > 20 else ''}
        - Data Types: {json.dumps(data_summary['data_types'], indent=2)}
        
        Numeric Summary:
        {json.dumps(data_summary.get('numeric_summary', {}), indent=2)}
        
        Categorical Summary:
        {json.dumps(data_summary.get('categorical_summary', {}), indent=2)}
        
        Please provide insights in JSON format with these keys:
        - key_findings: Array of 3-5 most important findings
        - recommendations: Array of 3-5 actionable recommendations
        - data_insights: Array of objects with 'title', 'description', and optional 'metric' (name, value)
        - data_quality_assessment: Object with quality score (1-10) and issues found
        - correlations_identified: Array of notable correlations or relationships found
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert data analyst specializing in business intelligence and data insights. Provide detailed, actionable insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _generate_creative_insights(self, data_summary: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate creative industry specific insights."""
        
        creative_cols = data_summary.get('potential_creative_metrics', [])
        
        if not creative_cols:
            return {'creative_metrics': {}, 'creative_recommendations': []}
        
        # Calculate creative metrics
        creative_metrics = {}
        for col in creative_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                creative_metrics[col] = {
                    'mean': float(df[col].mean()) if not df[col].empty else 0,
                    'median': float(df[col].median()) if not df[col].empty else 0,
                    'total': float(df[col].sum()) if not df[col].empty else 0,
                    'max': float(df[col].max()) if not df[col].empty else 0
                }
        
        if not creative_metrics:
            return {'creative_metrics': {}, 'creative_recommendations': []}
        
        prompt = f"""
        As a creative industry analytics expert, analyze these creative performance metrics:
        
        Creative Metrics Data:
        {json.dumps(creative_metrics, indent=2)}
        
        Dataset has {data_summary['shape']['rows']} records across {data_summary['shape']['columns']} dimensions.
        
        Time series data available: {bool(data_summary.get('time_series_columns'))}
        
        Provide creative-specific insights in JSON format with these keys:
        - creative_performance_analysis: Overall performance assessment
        - top_performing_metrics: Array of best performing metrics with explanations
        - optimization_opportunities: Array of specific areas for improvement
        - creative_recommendations: Array of actionable creative strategy recommendations
        - industry_benchmarks: Comparison to typical industry performance where applicable
        - engagement_patterns: Analysis of engagement-related metrics if available
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative industry analytics expert specializing in social media, marketing, and creative performance analysis. Provide actionable insights for creative professionals."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1200
            )
            
            creative_insights = json.loads(response.choices[0].message.content)
            creative_insights['creative_metrics'] = creative_metrics
            
            return creative_insights
            
        except Exception as e:
            return {
                'creative_metrics': creative_metrics,
                'creative_recommendations': [f"Error generating creative insights: {str(e)}"]
            }
    
    def explain_column(self, df: pd.DataFrame, column_name: str) -> str:
        """Generate AI explanation for a specific column."""
        if not self.client:
            return "OpenAI API not configured"
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        col_data = df[column_name]
        
        # Prepare column summary
        summary = {
            'name': column_name,
            'type': str(col_data.dtype),
            'non_null_count': int(col_data.count()),
            'null_count': int(col_data.isnull().sum()),
            'unique_count': int(col_data.nunique())
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            summary.update({
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'quartiles': col_data.quantile([0.25, 0.5, 0.75]).to_dict()
            })
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            value_counts = col_data.value_counts().head(10).to_dict()
            summary['top_values'] = value_counts
        
        prompt = f"""
        Explain this data column in simple terms for a creative professional:
        
        Column Details: {json.dumps(summary, indent=2)}
        
        Sample values: {col_data.dropna().head(10).tolist()}
        
        Provide a clear, non-technical explanation of what this column represents, 
        its significance for creative work, and any notable patterns or insights.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst who explains data in simple, creative-friendly terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def generate_story(self, insights: Dict[str, Any]) -> str:
        """Generate a narrative story from insights."""
        if not self.client:
            return "OpenAI API not configured"
        
        prompt = f"""
        Create a compelling data story from these insights for a creative professional:
        
        Insights: {json.dumps(insights, indent=2)}
        
        Write a narrative that:
        1. Starts with the most important finding
        2. Explains what the data reveals about creative performance
        3. Provides clear next steps and recommendations
        4. Uses accessible language and creative industry terminology
        5. Maintains an encouraging and actionable tone
        
        Keep it concise but engaging (3-4 paragraphs).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative consultant who turns data insights into compelling stories for creative professionals."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating story: {str(e)}"
