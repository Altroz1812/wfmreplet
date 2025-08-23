import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional
import seaborn as sns

class Visualizations:
    """Create interactive visualizations using Plotly."""
    
    def __init__(self):
        # Color palette for consistent styling
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Default layout settings
        self.default_layout = dict(
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=60, b=40)
        )
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create an interactive correlation heatmap."""
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
            **self.default_layout
        )
        
        return fig
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create distribution plot with histogram and box plot."""
        col_data = df[column].dropna()
        
        # Create subplot with histogram and box plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{column} Distribution', f'{column} Box Plot'],
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=col_data,
                nbinsx=30,
                name='Distribution',
                marker_color=self.color_palette[0],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=col_data,
                name='Box Plot',
                marker_color=self.color_palette[1],
                boxpoints='outliers'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Distribution Analysis: {column}",
            showlegend=False,
            **self.default_layout
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
        """Create interactive scatter plot."""
        if color_col:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=f"{y_col} vs {x_col}",
                color_discrete_sequence=self.color_palette,
                hover_data=[col for col in df.columns if col not in [x_col, y_col, color_col]][:3]
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=f"{y_col} vs {x_col}",
                color_discrete_sequence=[self.color_palette[0]],
                hover_data=[col for col in df.columns if col not in [x_col, y_col]][:3]
            )
        
        # Add trendline
        if len(df[x_col].dropna()) > 2 and len(df[y_col].dropna()) > 2:
            from sklearn.linear_model import LinearRegression
            
            # Fit linear regression for trendline
            clean_data = df[[x_col, y_col]].dropna()
            if len(clean_data) > 1:
                X = clean_data[x_col].values.reshape(-1, 1)
                y = clean_data[y_col].values
                
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    x_range = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
                    y_pred = model.predict(x_range.reshape(-1, 1))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_pred,
                            mode='lines',
                            name='Trendline',
                            line=dict(color='red', dash='dash')
                        )
                    )
                except:
                    pass  # Skip trendline if regression fails
        
        fig.update_layout(**self.default_layout)
        
        return fig
    
    def create_bar_chart(self, df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
        """Create interactive bar chart."""
        # Aggregate data
        agg_data = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'sum']).reset_index()
        agg_data.columns = [cat_col, 'Mean', 'Count', 'Sum']
        
        # Create bar chart with mean values
        fig = go.Figure(data=[
            go.Bar(
                x=agg_data[cat_col],
                y=agg_data['Mean'],
                marker_color=self.color_palette[0],
                hovertemplate="<b>%{x}</b><br>Average: %{y:.2f}<br>Count: %{customdata[0]}<br>Total: %{customdata[1]:.2f}<extra></extra>",
                customdata=np.column_stack((agg_data['Count'], agg_data['Sum']))
            )
        ])
        
        fig.update_layout(
            title=f"Average {num_col} by {cat_col}",
            xaxis_title=cat_col,
            yaxis_title=f"Average {num_col}",
            **self.default_layout
        )
        
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
        """Create interactive box plot."""
        fig = px.box(
            df, x=cat_col, y=num_col,
            title=f"{num_col} Distribution by {cat_col}",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(**self.default_layout)
        
        return fig
    
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
        """Create interactive histogram."""
        col_data = df[column].dropna()
        
        fig = go.Figure(data=[
            go.Histogram(
                x=col_data,
                nbinsx=bins,
                marker_color=self.color_palette[0],
                opacity=0.7,
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
            )
        ])
        
        # Add statistical annotations
        mean_val = col_data.mean()
        median_val = col_data.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green", annotation_text=f"Median: {median_val:.2f}")
        
        fig.update_layout(
            title=f"Histogram: {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            **self.default_layout
        )
        
        return fig
    
    def create_time_series_chart(self, df: pd.DataFrame, time_col: str, value_col: str) -> go.Figure:
        """Create interactive time series chart."""
        # Sort by time column
        df_sorted = df.sort_values(time_col)
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        
        # Create time series plot
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(
            go.Scatter(
                x=df_sorted[time_col],
                y=df_sorted[value_col],
                mode='lines+markers',
                name=value_col,
                line=dict(color=self.color_palette[0], width=2),
                marker=dict(size=4),
                hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>"
            )
        )
        
        # Add moving average if enough data points
        if len(df_sorted) >= 7:
            window = min(7, len(df_sorted) // 3)
            moving_avg = df_sorted[value_col].rolling(window=window, center=True).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[time_col],
                    y=moving_avg,
                    mode='lines',
                    name=f'{window}-period Moving Average',
                    line=dict(color=self.color_palette[1], width=2, dash='dash'),
                    hovertemplate="<b>%{x}</b><br>Moving Avg: %{y:.2f}<extra></extra>"
                )
            )
        
        fig.update_layout(
            title=f"Time Series: {value_col}",
            xaxis_title="Time",
            yaxis_title=value_col,
            hovermode='x unified',
            **self.default_layout
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=30, label="30D", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def create_missing_values_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing missing value patterns."""
        # Calculate missing values per column
        missing_data = df.isnull()
        
        if missing_data.sum().sum() == 0:
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found in the dataset!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Missing Values Analysis", **self.default_layout)
            return fig
        
        # Create heatmap for missing values pattern
        # Sample data if too large
        if len(df) > 1000:
            sample_indices = np.random.choice(len(df), 1000, replace=False)
            sample_indices.sort()
            missing_sample = missing_data.iloc[sample_indices]
        else:
            missing_sample = missing_data
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_sample.values.astype(int),
            x=missing_sample.columns,
            y=list(range(len(missing_sample))),
            colorscale=[[0, 'lightblue'], [1, 'red']],
            showscale=True,
            colorbar=dict(
                title="Missing",
                tickvals=[0, 1],
                ticktext=['Present', 'Missing']
            ),
            hovertemplate="<b>Column:</b> %{x}<br><b>Row:</b> %{y}<br><b>Status:</b> %{customdata}<extra></extra>",
            customdata=np.where(missing_sample.values, 'Missing', 'Present')
        ))
        
        fig.update_layout(
            title="Missing Values Pattern",
            xaxis_title="Columns",
            yaxis_title="Row Index",
            **self.default_layout
        )
        
        return fig
    
    def create_creative_dashboard(self, df: pd.DataFrame, metrics: List[str]) -> go.Figure:
        """Create a comprehensive dashboard for creative metrics."""
        # Create subplot layout
        n_metrics = len(metrics)
        rows = (n_metrics + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in metrics],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = self.color_palette[:n_metrics]
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                # Create histogram for each metric
                fig.add_trace(
                    go.Histogram(
                        x=df[metric].dropna(),
                        name=metric,
                        marker_color=colors[i],
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Creative Metrics Dashboard",
            showlegend=False,
            height=300 * rows,
            **self.default_layout
        )
        
        return fig
    
    def create_performance_comparison(self, df: pd.DataFrame, group_col: str, metrics: List[str]) -> go.Figure:
        """Create performance comparison chart across different groups."""
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in data")
        
        # Aggregate metrics by group
        agg_data = df.groupby(group_col)[metrics].mean().reset_index()
        
        # Create grouped bar chart
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=agg_data[group_col],
                    y=agg_data[metric],
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    hovertemplate=f"<b>{metric}</b><br>Group: %{{x}}<br>Average: %{{y:.2f}}<extra></extra>"
                )
            )
        
        fig.update_layout(
            title=f"Performance Comparison by {group_col.replace('_', ' ').title()}",
            xaxis_title=group_col.replace('_', ' ').title(),
            yaxis_title="Average Value",
            barmode='group',
            **self.default_layout
        )
        
        return fig
    
    def create_funnel_chart(self, df: pd.DataFrame, stage_col: str, value_col: str) -> go.Figure:
        """Create funnel chart for conversion analysis."""
        # Aggregate data by stage
        funnel_data = df.groupby(stage_col)[value_col].sum().reset_index()
        funnel_data = funnel_data.sort_values(value_col, ascending=False)
        
        fig = go.Figure(go.Funnel(
            y=funnel_data[stage_col],
            x=funnel_data[value_col],
            textinfo="value+percent initial",
            marker=dict(
                color=self.color_palette[:len(funnel_data)],
                line=dict(width=2, color="white")
            ),
            connector=dict(line=dict(color="royalblue", dash="dot", width=3))
        ))
        
        fig.update_layout(
            title=f"Conversion Funnel: {value_col.replace('_', ' ').title()}",
            **self.default_layout
        )
        
        return fig
