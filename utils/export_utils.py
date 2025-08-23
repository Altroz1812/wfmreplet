import pandas as pd
import json
import io
from datetime import datetime
from typing import Dict, Any, Optional
import base64

class ExportUtils:
    """Handle data and insights export functionality."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def to_csv(self, df: pd.DataFrame) -> str:
        """Export DataFrame to CSV format."""
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        return csv_buffer.getvalue()
    
    def to_excel(self, df: pd.DataFrame) -> bytes:
        """Export DataFrame to Excel format."""
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Summary statistics sheet
            if len(df.select_dtypes(include=['number']).columns) > 0:
                summary = df.describe()
                summary.to_excel(writer, sheet_name='Summary Statistics')
            
            # Data types sheet
            data_types = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            data_types.to_excel(writer, sheet_name='Data Info', index=False)
        
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    
    def to_json(self, df: pd.DataFrame) -> str:
        """Export DataFrame to JSON format."""
        # Convert DataFrame to JSON with proper handling of datetime and NaN
        json_str = df.to_json(orient='records', date_format='iso', indent=2)
        return json_str
    
    def export_insights_json(self, insights: Optional[Dict[str, Any]], trends: Optional[Dict[str, Any]]) -> str:
        """Export insights and trends to JSON format."""
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'format': 'insights_json',
                'version': '1.0'
            },
            'insights': insights or {},
            'trends': trends or {}
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def export_text_report(self, insights: Optional[Dict[str, Any]], trends: Optional[Dict[str, Any]]) -> str:
        """Export insights and trends as a formatted text report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("CREATIVE ANALYTICS AI - INSIGHTS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Data Overview
        if insights and 'data_overview' in insights:
            overview = insights['data_overview']
            report_lines.append("DATA OVERVIEW")
            report_lines.append("-" * 40)
            report_lines.append(f"Dataset Size: {overview.get('shape', {}).get('rows', 'N/A')} rows × {overview.get('shape', {}).get('columns', 'N/A')} columns")
            
            if 'columns' in overview:
                report_lines.append(f"Columns: {', '.join(overview['columns'][:10])}{'...' if len(overview['columns']) > 10 else ''}")
            report_lines.append("")
        
        # Key Findings
        if insights and 'key_findings' in insights:
            report_lines.append("KEY FINDINGS")
            report_lines.append("-" * 40)
            for i, finding in enumerate(insights['key_findings'], 1):
                report_lines.append(f"{i}. {finding}")
            report_lines.append("")
        
        # Recommendations
        if insights and 'recommendations' in insights:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for i, rec in enumerate(insights['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Creative Metrics
        if insights and 'creative_metrics' in insights:
            report_lines.append("CREATIVE METRICS")
            report_lines.append("-" * 40)
            for metric, values in insights['creative_metrics'].items():
                if isinstance(values, dict):
                    report_lines.append(f"{metric.replace('_', ' ').title()}:")
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  {key.replace('_', ' ').title()}: {value:,.2f}")
                        else:
                            report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
                else:
                    report_lines.append(f"{metric.replace('_', ' ').title()}: {values}")
            report_lines.append("")
        
        # Trends Analysis
        if trends:
            report_lines.append("TRENDS ANALYSIS")
            report_lines.append("-" * 40)
            
            # Summary
            if 'summary' in trends:
                summary = trends['summary']
                report_lines.append(f"Total Trends Detected: {summary.get('total_trends_detected', 0)}")
                report_lines.append(f"Strong Trends: {summary.get('strong_trends', 0)}")
                report_lines.append(f"Significant Correlations: {summary.get('significant_correlations', 0)}")
                report_lines.append(f"Patterns Found: {summary.get('patterns_found', 0)}")
                report_lines.append(f"Anomalies Detected: {summary.get('anomalies_detected', 0)}")
                report_lines.append("")
                
                if 'most_significant_findings' in summary:
                    report_lines.append("Most Significant Findings:")
                    for finding in summary['most_significant_findings']:
                        report_lines.append(f"• {finding}")
                    report_lines.append("")
            
            # Individual Trends
            if 'trends' in trends and trends['trends']:
                report_lines.append("DETECTED TRENDS")
                report_lines.append("-" * 30)
                for trend in trends['trends']:
                    report_lines.append(f"Column: {trend.get('column', 'Unknown')}")
                    report_lines.append(f"  Type: {trend.get('trend_type', 'Unknown')}")
                    report_lines.append(f"  Strength: {trend.get('strength', 'Unknown')}")
                    report_lines.append(f"  Description: {trend.get('description', 'No description')}")
                    if 'percentage_change' in trend:
                        report_lines.append(f"  Change: {trend['percentage_change']:.1f}%")
                    report_lines.append("")
            
            # Correlations
            if 'correlations' in trends and trends['correlations']:
                report_lines.append("SIGNIFICANT CORRELATIONS")
                report_lines.append("-" * 30)
                for corr in trends['correlations']:
                    report_lines.append(f"{corr.get('column_1', 'Unknown')} ↔ {corr.get('column_2', 'Unknown')}")
                    report_lines.append(f"  Correlation: {corr.get('correlation', 0):.3f}")
                    report_lines.append(f"  Type: {corr.get('type', 'Unknown')} ({corr.get('strength', 'Unknown')})")
                    report_lines.append("")
        
        # Data Quality Assessment
        if insights and 'data_quality_assessment' in insights:
            report_lines.append("DATA QUALITY ASSESSMENT")
            report_lines.append("-" * 40)
            quality = insights['data_quality_assessment']
            if 'quality_score' in quality:
                report_lines.append(f"Overall Quality Score: {quality['quality_score']}/10")
            if 'issues_found' in quality:
                report_lines.append("Issues Found:")
                for issue in quality['issues_found']:
                    report_lines.append(f"• {issue}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("Report generated by Creative Analytics AI")
        report_lines.append("Powered by OpenAI GPT-4o")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def create_summary_report(self, df: pd.DataFrame, insights: Optional[Dict[str, Any]] = None, trends: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a comprehensive summary report."""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_shape': {'rows': len(df), 'columns': len(df.columns)},
                'column_names': list(df.columns),
                'export_format': 'summary_report'
            },
            'data_summary': self._generate_data_summary(df),
            'insights_summary': self._summarize_insights(insights) if insights else None,
            'trends_summary': self._summarize_trends(trends) if trends else None
        }
        
        return report
    
    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary for export."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        
        summary = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'datetime_columns': len(datetime_cols),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'data_quality': {
                'missing_values_total': int(df.isnull().sum().sum()),
                'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'duplicate_rows': int(df.duplicated().sum()),
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            }
        }
        
        # Numeric summary
        if len(numeric_cols) > 0:
            numeric_summary = df[numeric_cols].describe()
            summary['numeric_statistics'] = numeric_summary.to_dict()
        
        # Categorical summary
        if len(categorical_cols) > 0:
            categorical_info = {}
            for col in categorical_cols:
                categorical_info[col] = {
                    'unique_count': int(df[col].nunique()),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'top_5_values': df[col].value_counts().head(5).to_dict()
                }
            summary['categorical_analysis'] = categorical_info
        
        return summary
    
    def _summarize_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize insights for export."""
        summary = {
            'key_findings_count': len(insights.get('key_findings', [])),
            'recommendations_count': len(insights.get('recommendations', [])),
            'insights_generated': bool(insights.get('key_findings') or insights.get('recommendations'))
        }
        
        # Extract key metrics
        if 'creative_metrics' in insights:
            summary['creative_metrics_available'] = True
            summary['creative_metrics_count'] = len(insights['creative_metrics'])
        
        if 'data_quality_assessment' in insights:
            quality = insights['data_quality_assessment']
            summary['data_quality_score'] = quality.get('quality_score')
        
        return summary
    
    def _summarize_trends(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize trends for export."""
        summary = trends.get('summary', {})
        
        return {
            'trends_detected': summary.get('total_trends_detected', 0),
            'strong_trends': summary.get('strong_trends', 0),
            'correlations_found': summary.get('significant_correlations', 0),
            'patterns_identified': summary.get('patterns_found', 0),
            'anomalies_detected': summary.get('anomalies_detected', 0),
            'time_series_analysis': summary.get('time_series_available', False)
        }
    
    def export_visualization_data(self, df: pd.DataFrame, chart_type: str, **kwargs) -> Dict[str, Any]:
        """Export data formatted for specific visualization types."""
        export_data = {
            'chart_type': chart_type,
            'timestamp': datetime.now().isoformat(),
            'data': {},
            'metadata': {
                'total_rows': len(df),
                'columns': list(df.columns)
            }
        }
        
        if chart_type == 'correlation_heatmap':
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                export_data['data'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'columns': list(numeric_cols)
                }
        
        elif chart_type == 'time_series':
            time_col = kwargs.get('time_col')
            value_col = kwargs.get('value_col')
            if time_col and value_col:
                ts_data = df[[time_col, value_col]].dropna().sort_values(time_col)
                export_data['data'] = {
                    'time_series': ts_data.to_dict('records'),
                    'time_column': time_col,
                    'value_column': value_col
                }
        
        return export_data
