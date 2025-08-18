"""
Statistics and metadata panel component
"""

import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import numpy as np

def create_statistics_panel():
    """Create the statistics and metadata panel"""
    
    stats_panel = dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-chart-bar me-2"),
                "File Statistics"
            ], className="mb-0")
        ]),
        
        dbc.CardBody([
            html.Div(id='statistics-content', children=[
                html.P("Upload a file to see statistics", className="text-muted text-center")
            ])
        ])
    ])
    
    return stats_panel

def generate_statistics(df, file_type, metadata=None):
    """Generate statistics for uploaded data"""
    
    if df is None or df.empty:
        return html.P("No data available", className="text-muted")
    
    stats_items = []
    
    # Basic info
    stats_items.extend([
        create_stat_item("Rows", f"{len(df):,}"),
        create_stat_item("Columns", f"{len(df.columns):,}"),
        html.Hr()
    ])
    
    if file_type == "csv":
        # CSV-specific statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_items.extend([
            create_stat_item("Numeric Columns", f"{len(numeric_cols)}"),
            create_stat_item("Text Columns", f"{len(df.columns) - len(numeric_cols)}"),
            create_stat_item("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB"),
        ])
        
        if len(numeric_cols) > 0:
            stats_items.extend([
                html.Hr(),
                html.H6("Numeric Summary", className="fw-bold"),
                html.Pre(df[numeric_cols].describe().round(2).to_string(), 
                        style={'font-size': '0.8rem', 'background': '#f8f9fa', 'padding': '10px'})
            ])
    
    elif file_type == "fits":
        # FITS-specific statistics
        if metadata:
            stats_items.extend([
                create_stat_item("HDUs", f"{metadata.get('n_hdus', 'N/A')}"),
                create_stat_item("Data Type", f"{metadata.get('data_type', 'N/A')}"),
                create_stat_item("Dimensions", f"{metadata.get('dimensions', 'N/A')}"),
                html.Hr()
            ])
            
        # Array statistics for FITS data
        if hasattr(df, 'values') and df.values.size > 0:
            data_array = df.values
            stats_items.extend([
                create_stat_item("Min Value", f"{np.nanmin(data_array):.3e}"),
                create_stat_item("Max Value", f"{np.nanmax(data_array):.3e}"),
                create_stat_item("Mean", f"{np.nanmean(data_array):.3e}"),
                create_stat_item("Std Dev", f"{np.nanstd(data_array):.3e}"),
            ])
    
    return html.Div(stats_items)

def create_stat_item(label, value):
    """Create a statistics item"""
    return dbc.Row([
        dbc.Col(html.Strong(label), width=6),
        dbc.Col(html.Span(value, className="text-primary"), width=6)
    ], className="mb-2")
