import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class TrendDetector:
    """Advanced trend detection and pattern analysis."""
    
    def __init__(self):
        self.trend_threshold = 0.1  # Minimum correlation for trend detection
        self.seasonality_periods = [7, 30, 90, 365]  # Common business periods
    
    def detect_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive trend detection across the dataset."""
        
        trends_report = {
            'trends': [],
            'patterns': [],
            'correlations': [],
            'time_series_analysis': {},
            'anomalies': [],
            'summary': {}
        }
        
        try:
            # 1. Detect linear trends in numeric columns
            numeric_trends = self._detect_linear_trends(df)
            trends_report['trends'].extend(numeric_trends)
            
            # 2. Detect correlation patterns
            correlation_patterns = self._detect_correlations(df)
            trends_report['correlations'].extend(correlation_patterns)
            
            # 3. Time series analysis if applicable
            time_series_analysis = self._analyze_time_series(df)
            trends_report['time_series_analysis'] = time_series_analysis
            
            # 4. Pattern detection
            patterns = self._detect_patterns(df)
            trends_report['patterns'].extend(patterns)
            
            # 5. Anomaly detection
            anomalies = self._detect_anomalies(df)
            trends_report['anomalies'].extend(anomalies)
            
            # 6. Generate summary
            trends_report['summary'] = self._generate_trend_summary(trends_report)
            
        except Exception as e:
            trends_report['error'] = f"Error in trend detection: {str(e)}"
        
        return trends_report
    
    def _detect_linear_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect linear trends in numeric columns."""
        trends = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() / len(df) > 0.5:  # Skip columns with >50% missing
                continue
            
            # Remove NaN values
            clean_data = df[col].dropna()
            if len(clean_data) < 10:  # Need minimum data points
                continue
            
            # Create time index for trend analysis
            x = np.arange(len(clean_data)).reshape(-1, 1)
            y = clean_data.values
            
            try:
                # Fit linear regression
                model = LinearRegression()
                model.fit(x, y)
                
                # Calculate trend metrics
                slope = model.coef_[0]
                r_score = model.score(x, y)
                
                # Determine trend direction and strength
                if abs(slope) > self.trend_threshold and r_score > 0.1:
                    trend_type = "Increasing" if slope > 0 else "Decreasing"
                    strength = "Strong" if r_score > 0.7 else "Moderate" if r_score > 0.3 else "Weak"
                    
                    # Calculate percentage change
                    start_val = y[0] if len(y) > 0 else 0
                    end_val = y[-1] if len(y) > 0 else 0
                    pct_change = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                    
                    trends.append({
                        'column': col,
                        'trend_type': trend_type,
                        'strength': strength,
                        'slope': float(slope),
                        'correlation': float(r_score),
                        'percentage_change': float(pct_change),
                        'description': f"{col} shows a {strength.lower()} {trend_type.lower()} trend with {pct_change:.1f}% change overall"
                    })
            
            except Exception as e:
                continue  # Skip problematic columns
        
        return trends
    
    def _detect_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect significant correlations between numeric columns."""
        correlations = []
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return correlations
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find significant correlations
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i >= j:  # Avoid duplicates and self-correlation
                    continue
                
                corr_value = corr_matrix.loc[col1, col2]
                
                if abs(corr_value) > 0.5:  # Significant correlation threshold
                    correlation_type = "Positive" if corr_value > 0 else "Negative"
                    strength = "Very Strong" if abs(corr_value) > 0.8 else "Strong" if abs(corr_value) > 0.6 else "Moderate"
                    
                    correlations.append({
                        'column_1': col1,
                        'column_2': col2,
                        'correlation': float(corr_value),
                        'type': correlation_type,
                        'strength': strength,
                        'description': f"{strength} {correlation_type.lower()} correlation between {col1} and {col2} (r={corr_value:.3f})"
                    })
        
        return correlations
    
    def _analyze_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series patterns if temporal data exists."""
        analysis = {}
        
        # Try to find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        # Also check for columns that might be dates
        for col in df.columns:
            if col not in datetime_cols:
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        if not datetime_cols:
            return analysis
        
        primary_time_col = datetime_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return analysis
        
        try:
            # Sort by time column
            df_sorted = df.sort_values(primary_time_col)
            
            for metric_col in numeric_cols[:5]:  # Analyze top 5 numeric columns
                if df_sorted[metric_col].isnull().sum() / len(df_sorted) > 0.7:
                    continue
                
                # Basic time series metrics
                ts_analysis = {
                    'column': metric_col,
                    'time_span': {
                        'start': str(df_sorted[primary_time_col].min()),
                        'end': str(df_sorted[primary_time_col].max()),
                        'duration_days': (df_sorted[primary_time_col].max() - df_sorted[primary_time_col].min()).days
                    }
                }
                
                # Detect seasonality patterns
                seasonality = self._detect_seasonality(df_sorted, primary_time_col, metric_col)
                if seasonality:
                    ts_analysis['seasonality'] = seasonality
                
                # Growth rate analysis
                growth_analysis = self._analyze_growth_rate(df_sorted, primary_time_col, metric_col)
                if growth_analysis:
                    ts_analysis['growth'] = growth_analysis
                
                analysis[metric_col] = ts_analysis
        
        except Exception as e:
            analysis['error'] = f"Time series analysis error: {str(e)}"
        
        return analysis
    
    def _detect_seasonality(self, df: pd.DataFrame, time_col: str, value_col: str) -> Optional[Dict[str, Any]]:
        """Detect seasonal patterns in time series data."""
        try:
            # Create time-based features
            df_clean = df[[time_col, value_col]].dropna()
            if len(df_clean) < 30:  # Need sufficient data
                return None
            
            df_clean[time_col] = pd.to_datetime(df_clean[time_col])
            df_clean['day_of_week'] = df_clean[time_col].dt.dayofweek
            df_clean['day_of_month'] = df_clean[time_col].dt.day
            df_clean['month'] = df_clean[time_col].dt.month
            
            seasonality_patterns = {}
            
            # Weekly pattern
            weekly_pattern = df_clean.groupby('day_of_week')[value_col].mean()
            weekly_var = weekly_pattern.var()
            if weekly_var > 0:
                seasonality_patterns['weekly'] = {
                    'variance': float(weekly_var),
                    'pattern': weekly_pattern.to_dict()
                }
            
            # Monthly pattern
            if df_clean['month'].nunique() > 3:  # Need data across multiple months
                monthly_pattern = df_clean.groupby('month')[value_col].mean()
                monthly_var = monthly_pattern.var()
                if monthly_var > 0:
                    seasonality_patterns['monthly'] = {
                        'variance': float(monthly_var),
                        'pattern': monthly_pattern.to_dict()
                    }
            
            return seasonality_patterns if seasonality_patterns else None
            
        except Exception:
            return None
    
    def _analyze_growth_rate(self, df: pd.DataFrame, time_col: str, value_col: str) -> Optional[Dict[str, Any]]:
        """Analyze growth rate patterns in time series."""
        try:
            df_clean = df[[time_col, value_col]].dropna().sort_values(time_col)
            if len(df_clean) < 10:
                return None
            
            values = df_clean[value_col].values
            
            # Calculate period-over-period growth rates
            growth_rates = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    growth_rate = (values[i] - values[i-1]) / values[i-1]
                    growth_rates.append(growth_rate)
            
            if not growth_rates:
                return None
            
            growth_rates = np.array(growth_rates)
            
            return {
                'average_growth_rate': float(np.mean(growth_rates)),
                'growth_volatility': float(np.std(growth_rates)),
                'positive_periods': int(np.sum(growth_rates > 0)),
                'negative_periods': int(np.sum(growth_rates < 0)),
                'max_growth': float(np.max(growth_rates)),
                'max_decline': float(np.min(growth_rates))
            }
            
        except Exception:
            return None
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect various data patterns."""
        patterns = []
        
        # Pattern 1: Outlier concentration
        outlier_pattern = self._detect_outlier_patterns(df)
        if outlier_pattern:
            patterns.append(outlier_pattern)
        
        # Pattern 2: Distribution patterns
        distribution_patterns = self._analyze_distributions(df)
        patterns.extend(distribution_patterns)
        
        # Pattern 3: Missing value patterns
        missing_pattern = self._analyze_missing_patterns(df)
        if missing_pattern:
            patterns.append(missing_pattern)
        
        return patterns
    
    def _detect_outlier_patterns(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect patterns in outliers."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        outlier_info = {}
        total_outliers = 0
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(col_data)) * 100
                }
                total_outliers += outlier_count
        
        if total_outliers > 0:
            return {
                'type': 'Outlier Distribution',
                'description': f"Found {total_outliers} outliers across {len(outlier_info)} columns",
                'details': outlier_info,
                'confidence': min(1.0, total_outliers / len(df))
            }
        
        return None
    
    def _analyze_distributions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze distribution patterns in numeric columns."""
        patterns = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 20:
                continue
            
            try:
                # Test for normality
                _, p_value = stats.normaltest(col_data)
                
                # Calculate skewness and kurtosis
                skewness = stats.skew(col_data)
                kurtosis = stats.kurtosis(col_data)
                
                distribution_type = "Normal"
                if abs(skewness) > 1:
                    distribution_type = "Highly Skewed"
                elif abs(skewness) > 0.5:
                    distribution_type = "Moderately Skewed"
                
                if abs(kurtosis) > 3:
                    distribution_type += " with High Kurtosis"
                
                patterns.append({
                    'type': 'Distribution Pattern',
                    'column': col,
                    'distribution_type': distribution_type,
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'normality_p_value': float(p_value),
                    'description': f"{col} follows a {distribution_type.lower()} distribution pattern",
                    'confidence': 1.0 - p_value if p_value < 0.05 else p_value
                })
                
            except Exception:
                continue
        
        return patterns
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze patterns in missing data."""
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) == 0:
            return None
        
        total_missing = missing_info.sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (total_missing / total_cells) * 100
        
        # Check for systematic missing patterns
        systematic_missing = []
        for col in missing_cols.index:
            col_missing_pct = (missing_cols[col] / len(df)) * 100
            if col_missing_pct > 50:
                systematic_missing.append(col)
        
        return {
            'type': 'Missing Data Pattern',
            'description': f"Missing data affects {len(missing_cols)} columns ({missing_percentage:.1f}% of total data)",
            'total_missing_percentage': missing_percentage,
            'columns_affected': len(missing_cols),
            'systematic_missing_columns': systematic_missing,
            'confidence': min(1.0, missing_percentage / 100)
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in the data."""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            
            try:
                # Statistical anomaly detection using Z-score
                z_scores = np.abs(stats.zscore(col_data))
                anomaly_indices = np.where(z_scores > 3)[0]
                
                if len(anomaly_indices) > 0:
                    anomaly_values = col_data.iloc[anomaly_indices]
                    
                    anomalies.append({
                        'type': 'Statistical Anomaly',
                        'column': col,
                        'count': len(anomaly_indices),
                        'percentage': (len(anomaly_indices) / len(col_data)) * 100,
                        'description': f"Found {len(anomaly_indices)} statistical anomalies in {col}",
                        'sample_values': anomaly_values.head(3).tolist(),
                        'confidence': min(1.0, len(anomaly_indices) / len(col_data) * 10)
                    })
                    
            except Exception:
                continue
        
        return anomalies
    
    def _generate_trend_summary(self, trends_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all detected trends and patterns."""
        summary = {
            'total_trends_detected': len(trends_report.get('trends', [])),
            'strong_trends': len([t for t in trends_report.get('trends', []) if t.get('strength') == 'Strong']),
            'significant_correlations': len([c for c in trends_report.get('correlations', []) if abs(c.get('correlation', 0)) > 0.7]),
            'patterns_found': len(trends_report.get('patterns', [])),
            'anomalies_detected': len(trends_report.get('anomalies', [])),
            'time_series_available': bool(trends_report.get('time_series_analysis'))
        }
        
        # Identify most significant findings
        significant_findings = []
        
        for trend in trends_report.get('trends', []):
            if trend.get('strength') == 'Strong':
                significant_findings.append(f"Strong {trend.get('trend_type', '').lower()} trend in {trend.get('column')}")
        
        for corr in trends_report.get('correlations', []):
            if abs(corr.get('correlation', 0)) > 0.8:
                significant_findings.append(f"Very strong correlation between {corr.get('column_1')} and {corr.get('column_2')}")
        
        summary['most_significant_findings'] = significant_findings[:5]  # Top 5
        
        return summary
