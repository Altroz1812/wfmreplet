import pandas as pd
import numpy as np
import json
import io
import streamlit as st
from typing import Dict, List, Any, Optional

class DataProcessor:
    """Handles data loading, processing, and quality assessment."""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json']
    
    def process_file(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded file and return DataFrame."""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                content = uploaded_file.read().decode('utf-8')
                json_data = json.loads(content)
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    df = pd.json_normalize(json_data)
                elif isinstance(json_data, dict):
                    if len(json_data) == 1 and isinstance(list(json_data.values())[0], list):
                        # JSON with single key containing array
                        df = pd.json_normalize(list(json_data.values())[0])
                    else:
                        # Nested JSON object
                        df = pd.json_normalize(json_data)
                else:
                    raise ValueError("Invalid JSON structure")
            
            # Basic data cleaning
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error processing {file_extension} file: {str(e)}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations."""
        # Remove completely empty rows and columns
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.astype(str)
        df.columns = df.columns.str.strip()
        
        # Try to infer better data types
        df = self._infer_data_types(df)
        
        return df
    
    def _infer_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to infer better data types for columns."""
        for col in df.columns:
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Try numeric conversion
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.5:  # If >50% can be converted
                    df[col] = numeric_series
                    continue
                
                # Try datetime conversion
                try:
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    if datetime_series.notna().sum() > len(df) * 0.5:  # If >50% can be converted
                        df[col] = datetime_series
                        continue
                except:
                    pass
                
                # Try boolean conversion
                if set(df[col].str.lower().unique()) <= {'true', 'false', 'yes', 'no', '1', '0'}:
                    df[col] = df[col].str.lower().map({
                        'true': True, 'false': False, 'yes': True, 'no': False,
                        '1': True, '0': False
                    })
        
        return df
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and return metrics."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
            'duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        return quality_report
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get detailed information about each column."""
        column_info = []
        
        for col in df.columns:
            col_data = df[col]
            info = {
                'Column': col,
                'Data Type': str(col_data.dtype),
                'Non-Null Count': col_data.count(),
                'Null Count': col_data.isnull().sum(),
                'Null %': (col_data.isnull().sum() / len(df)) * 100,
                'Unique Values': col_data.nunique(),
                'Memory Usage (KB)': col_data.memory_usage(deep=True) / 1024
            }
            
            # Add type-specific info
            if pd.api.types.is_numeric_dtype(col_data):
                info.update({
                    'Min': col_data.min(),
                    'Max': col_data.max(),
                    'Mean': col_data.mean(),
                    'Std Dev': col_data.std()
                })
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                info.update({
                    'Min Date': col_data.min(),
                    'Max Date': col_data.max(),
                    'Date Range (Days)': (col_data.max() - col_data.min()).days if col_data.max() != col_data.min() else 0
                })
            else:
                # For categorical/text data
                info.update({
                    'Most Frequent': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else 'N/A',
                    'Avg Length': col_data.astype(str).str.len().mean() if not col_data.empty else 0
                })
            
            column_info.append(info)
        
        return pd.DataFrame(column_info)
    
    def identify_creative_metrics(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that might be creative/marketing metrics."""
        creative_keywords = [
            'engagement', 'click', 'ctr', 'conversion', 'view', 'impression',
            'reach', 'like', 'share', 'comment', 'follow', 'subscriber',
            'bounce', 'session', 'pageview', 'roi', 'revenue', 'cost',
            'cpm', 'cpc', 'cpa', 'roas', 'frequency', 'rating', 'score'
        ]
        
        potential_metrics = []
        
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace('-', '').replace(' ', '')
            for keyword in creative_keywords:
                if keyword in col_lower:
                    potential_metrics.append(col)
                    break
        
        return potential_metrics
    
    def identify_time_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain time/date data."""
        time_columns = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_columns.append(col)
            else:
                # Check if column name suggests time data
                time_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'when']
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in time_keywords):
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(df[col])
                        time_columns.append(col)
                    except:
                        pass
        
        return time_columns
    
    def identify_categorical_columns(self, df: pd.DataFrame, max_unique_ratio: float = 0.1) -> List[str]:
        """Identify categorical columns based on data types and unique value ratios."""
        categorical_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio <= max_unique_ratio:
                    categorical_columns.append(col)
            elif df[col].dtype == 'bool':
                categorical_columns.append(col)
        
        return categorical_columns
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(include=['object', 'category', 'bool'])
        datetime_df = df.select_dtypes(include=['datetime64[ns]'])
        
        summary = {
            'overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_df.columns),
                'categorical_columns': len(categorical_df.columns),
                'datetime_columns': len(datetime_df.columns)
            }
        }
        
        if not numeric_df.empty:
            summary['numeric_summary'] = numeric_df.describe().to_dict()
        
        if not categorical_df.empty:
            summary['categorical_summary'] = {}
            for col in categorical_df.columns:
                summary['categorical_summary'][col] = {
                    'unique_count': categorical_df[col].nunique(),
                    'most_frequent': categorical_df[col].mode().iloc[0] if len(categorical_df[col].mode()) > 0 else None,
                    'frequency': categorical_df[col].value_counts().head(5).to_dict()
                }
        
        if not datetime_df.empty:
            summary['datetime_summary'] = {}
            for col in datetime_df.columns:
                summary['datetime_summary'][col] = {
                    'min_date': str(datetime_df[col].min()),
                    'max_date': str(datetime_df[col].max()),
                    'date_range_days': (datetime_df[col].max() - datetime_df[col].min()).days
                }
        
        return summary
