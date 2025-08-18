"""

CSV Data Plotting Components for AstroAid

"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class CSVPlotter:
    """Advanced CSV plotting functionality"""
    
    def __init__(self):
        self.plot_types = {
            'line': 'Line Plot',
            'scatter': 'Scatter Plot', 
            'bar': 'Bar Chart',
            'histogram': 'Histogram',
            'box': 'Box Plot',
            'heatmap': 'Heatmap',
            'violin': 'Violin Plot',
            'density': 'Density Plot'
        }
    
    def create_plot_controls(self, columns: List[str]) -> html.Div:
        """Create interactive plot controls"""
        numeric_columns = self._get_numeric_columns(columns)
        
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-line me-2"),
                    "Interactive Plot Controls"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # Plot Type Selection
                dbc.Row([
                    dbc.Col([
                        html.Label("📊 Plot Type:", className="fw-bold"),
                        dcc.Dropdown(
                            id='plot-type-dropdown',
                            options=[{'label': v, 'value': k} for k, v in self.plot_types.items()],
                            value='line',
                            clearable=False
                        )
                    ], width=4),
                    
                    # X-Axis Selection
                    dbc.Col([
                        html.Label("📈 X-Axis:", className="fw-bold"),
                        dcc.Dropdown(
                            id='x-axis-dropdown',
                            options=[{'label': col, 'value': col} for col in columns],
                            value=columns[0] if columns else None,
                            clearable=False
                        )
                    ], width=4),
                    
                    # Y-Axis Selection
                    dbc.Col([
                        html.Label("📊 Y-Axis:", className="fw-bold"),
                        dcc.Dropdown(
                            id='y-axis-dropdown',
                            options=[{'label': col, 'value': col} for col in numeric_columns],
                            value=numeric_columns[0] if numeric_columns else None,
                            clearable=False
                        )
                    ], width=4)
                ], className="mb-3"),
                
                # Additional Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("🎨 Color By:", className="fw-bold"),
                        dcc.Dropdown(
                            id='color-by-dropdown',
                            options=[{'label': 'None', 'value': None}] + 
                                   [{'label': col, 'value': col} for col in columns],
                            value=None,
                            clearable=True,
                            placeholder="Select column for coloring"
                        )
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("📏 Size By (Scatter):", className="fw-bold"),
                        dcc.Dropdown(
                            id='size-by-dropdown',
                            options=[{'label': 'None', 'value': None}] + 
                                   [{'label': col, 'value': col} for col in numeric_columns],
                            value=None,
                            clearable=True,
                            placeholder="Select column for sizing"
                        )
                    ], width=6)
                ], className="mb-3"),
                
                # Advanced Options
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id='plot-options',
                            options=[
                                {'label': ' Log X-axis', 'value': 'log_x'},
                                {'label': ' Log Y-axis', 'value': 'log_y'},
                                {'label': ' Show trendline', 'value': 'trendline'},
                                {'label': ' Smooth lines', 'value': 'smooth'}
                            ],
                            value=[],
                            inline=True
                        )
                    ], width=12)
                ])
            ])
        ], className="mb-4")
    
    def _get_numeric_columns(self, columns: List[str]) -> List[str]:
        """Filter to numeric columns based on name patterns"""
        # This is a heuristic - in practice you'd check actual data types
        numeric_indicators = ['num', 'count', 'value', 'amount', 'price', 'rate', 'score', 'temp', 'pressure']
        numeric_cols = []
        
        for col in columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in numeric_indicators):
                numeric_cols.append(col)
            # Also include columns that likely contain numbers
            elif any(char.isdigit() for char in col):
                numeric_cols.append(col)
        
        # If no obvious numeric columns, return all (user can figure it out)
        return numeric_cols if numeric_cols else columns
    
    def create_plot(self, df: pd.DataFrame, plot_type: str, x_col: str, y_col: str, 
                   color_col: str = None, size_col: str = None, options: List[str] = None) -> go.Figure:
        """Create the actual plot based on parameters"""
        if df.empty or not x_col or not y_col:
            return go.Figure().add_annotation(text="No data to plot", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        options = options or []
        
        try:
            # Prepare common parameters
            plot_params = {
                'data_frame': df,
                'x': x_col,
                'title': f"{plot_type.title()} Plot: {y_col} vs {x_col}"
            }
            
            if y_col and y_col != x_col:
                plot_params['y'] = y_col
            
            if color_col:
                plot_params['color'] = color_col
            
            # Create plot based on type
            if plot_type == 'line':
                if 'smooth' in options:
                    plot_params['line_shape'] = 'spline'
                fig = px.line(**plot_params)
                if 'trendline' in options:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline="ols")
                    
            elif plot_type == 'scatter':
                if size_col:
                    plot_params['size'] = size_col
                fig = px.scatter(**plot_params)
                if 'trendline' in options:
                    plot_params['trendline'] = 'ols'
                    fig = px.scatter(**plot_params)
                    
            elif plot_type == 'bar':
                fig = px.bar(**plot_params)
                
            elif plot_type == 'histogram':
                plot_params.pop('y', None)  # Remove y for histogram
                fig = px.histogram(**plot_params, nbins=30)
                
            elif plot_type == 'box':
                fig = px.box(**plot_params)
                
            elif plot_type == 'violin':
                fig = px.violin(**plot_params)
                
            elif plot_type == 'density':
                plot_params.pop('y', None)
                fig = px.density_contour(**plot_params)
                
            elif plot_type == 'heatmap':
                # For heatmap, create correlation matrix if both columns are numeric
                try:
                    corr_matrix = df.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                   title="Correlation Heatmap")
                except:
                    fig = px.scatter(**plot_params)  # Fallback
            
            else:
                fig = px.line(**plot_params)  # Default fallback
            
            # Apply log scales if requested
            if 'log_x' in options:
                fig.update_xaxes(type="log")
            if 'log_y' in options:
                fig.update_yaxes(type="log")
            
            # Styling
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=16,
                showlegend=True if color_col else False,
                height=500
            )
            
            fig.update_xaxes(showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridcolor='lightgray')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return go.Figure().add_annotation(text=f"Error creating plot: {str(e)}", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)

def create_csv_plotting_interface() -> html.Div:
    """Create the main plotting interface"""
    return html.Div([
        html.Div(id='plot-controls-container'),
        dcc.Graph(id='csv-plot', style={'height': '600px'}),
        
        # Statistics Summary
        html.Div(id='plot-statistics', className="mt-4")
    ])

def register_plotting_callbacks(app, csv_plotter: CSVPlotter):
    """Register all plotting-related callbacks"""
    
    @app.callback(
        Output('plot-controls-container', 'children'),
        [Input('processed-data', 'data')]
    )
    def update_plot_controls(processed_data):
        """Update plot controls when new data is loaded"""
        if not processed_data or not processed_data.get('columns'):
            return html.Div("No data available for plotting", className="text-muted text-center py-4")
        
        columns = processed_data['columns']
        return csv_plotter.create_plot_controls(columns)
    
    @app.callback(
        Output('csv-plot', 'figure'),
        [Input('plot-type-dropdown', 'value'),
         Input('x-axis-dropdown', 'value'),
         Input('y-axis-dropdown', 'value'),
         Input('color-by-dropdown', 'value'),
         Input('size-by-dropdown', 'value'),
         Input('plot-options', 'value')],
        [State('processed-data', 'data')]
    )
    def update_plot(plot_type, x_col, y_col, color_col, size_col, options, processed_data):
        """Update the main plot based on user selections"""
        if not processed_data or not processed_data.get('data'):
            return go.Figure()
        
        df = pd.DataFrame(processed_data['data'])
        
        return csv_plotter.create_plot(
            df=df,
            plot_type=plot_type,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            size_col=size_col,
            options=options
        )
    
    @app.callback(
        Output('plot-statistics', 'children'),
        [Input('processed-data', 'data')]
    )
    def update_statistics(processed_data):
        """Show basic statistics about the data"""
        if not processed_data or not processed_data.get('data'):
            return html.Div()
        
        df = pd.DataFrame(processed_data['data'])
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return dbc.Alert("No numeric data available for statistics", color="info")
        
        stats = numeric_df.describe()
        
        return dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="fas fa-calculator me-2"),
                    "Data Statistics"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.Div([
                    html.Small(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns", 
                              className="text-muted"),
                    html.Br(),
                    html.Small(f"Numeric columns: {len(numeric_df.columns)}", 
                              className="text-muted")
                ])
            ])
        ])
