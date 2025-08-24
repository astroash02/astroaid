"""
Visualization callbacks - FIXED VERSION
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback, Input, Output, State, dash_table, html, ctx, no_update
import numpy as np
import logging

logger = logging.getLogger(__name__)

def register_callbacks(app):
    """Register all visualization callbacks"""
    
    @callback(
        [Output("x-data-selector", "options"),
         Output("y-data-selector", "options"),
         Output("color-by-selector", "options"),
         Output("size-by-selector", "options"),
         Output("quick-column-selector", "options"),
         Output("quick-column-selector", "value"),
         Output("quick-row-limit", "value")],
        [Input("processed-data", "data")]
    )
    def update_excel_selectors(stored_data):
        """Update selectors with smart defaults"""
        if not stored_data:
            return [], [], [], [], [], [], 100

        try:
            if isinstance(stored_data, dict) and 'data' in stored_data:
                df = pd.DataFrame(stored_data['data'])
            else:
                df = pd.DataFrame(stored_data)
        except Exception as e:
            logger.error(f"Error creating dataframe from stored_data: {e}")
            return [], [], [], [], [], [], 100

        if df.empty:
            return [], [], [], [], [], [], 100

        all_columns = [{"label": col, "value": col} for col in df.columns]
        numeric_columns = [{"label": col, "value": col} for col in df.select_dtypes(include=[np.number]).columns]

        # Smart defaults
        display_columns = [col for col in df.columns if col != '_source_file']
        default_columns = display_columns[:8] if len(display_columns) > 8 else display_columns

        # Row limit based on data size
        total_rows = len(df)
        if total_rows <= 50:
            default_row_limit = total_rows
        elif total_rows <= 500:
            default_row_limit = 100
        elif total_rows <= 5000:
            default_row_limit = 200
        else:
            default_row_limit = 500

        return (
            numeric_columns, numeric_columns, all_columns, numeric_columns,
            all_columns, default_columns, default_row_limit
        )

    @callback(
        Output("excel-style-chart", "figure"),
        [Input("create-chart-btn", "n_clicks")],
        [State("chart-type-selector", "value"),
         State("x-data-selector", "value"),
         State("y-data-selector", "value"),
         State("color-by-selector", "value"),
         State("size-by-selector", "value"),
         State("processed-data", "data")],
        prevent_initial_call=True
    )
    def update_chart(n_clicks, chart_type, x_col, y_col, color_col, size_col, data):
        """Create chart when button is clicked"""
        if not n_clicks or n_clicks == 0:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Select chart options above and click 'Create Chart' to generate visualization",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            empty_fig.update_layout(height=500, showlegend=False)
            return empty_fig

        if not data:
            error_fig = go.Figure()
            error_fig.add_annotation(
                text="No data available. Please upload a CSV file first.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            error_fig.update_layout(height=500, showlegend=False)
            return error_fig

        try:
            # Handle different data formats
            if isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)

            if df.empty:
                raise ValueError("Dataset is empty")

            if not x_col:
                raise ValueError("Please select a column for X-axis")

            if x_col not in df.columns:
                raise ValueError(f"Column '{x_col}' not found in data")

            if chart_type in ['scatter', 'line', 'bar'] and not y_col:
                raise ValueError(f"Please select a column for Y-axis for {chart_type} chart")

            # Create the plot
            if chart_type == 'scatter' and x_col and y_col:
                fig = px.scatter(
                    df, x=x_col, y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    size=size_col if size_col and size_col in df.columns else None,
                    title=f'Scatter Plot: {y_col} vs {x_col}'
                )
            elif chart_type == 'line' and x_col and y_col:
                fig = px.line(
                    df, x=x_col, y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f'Line Plot: {y_col} vs {x_col}'
                )
            elif chart_type == 'bar' and x_col and y_col:
                fig = px.bar(
                    df, x=x_col, y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f'Bar Chart: {y_col} vs {x_col}'
                )
            elif chart_type == 'histogram' and x_col:
                fig = px.histogram(
                    df, x=x_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f'Histogram of {x_col}',
                    nbins=30
                )
            elif chart_type == 'box' and y_col:
                fig = px.box(
                    df, y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f'Box Plot of {y_col}'
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Please select appropriate columns for {chart_type or 'selected'} chart",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="orange")
                )

            # Apply styling
            fig.update_layout(
                height=500,
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                margin=dict(l=60, r=60, t=80, b=60),
                showlegend=True if color_col and color_col in df.columns else False,
                title_font_size=16,
                title_x=0.5
            )

            fig.update_xaxes(showgrid=True, gridcolor='lightgray', title_font_size=14)
            fig.update_yaxes(showgrid=True, gridcolor='lightgray', title_font_size=14)

            logger.info(f"Successfully created {chart_type} chart!")
            return fig

        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f'Error: {str(e)}',
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            error_fig.update_layout(height=500, showlegend=False)
            return error_fig

    @callback(
        Output("main-data-table", "children"),
        [Input("quick-column-selector", "value"),
         Input("quick-row-limit", "value"),
         Input("refresh-btn", "n_clicks")],
        [State("processed-data", "data")]
    )
    def update_combined_table(selected_columns, row_limit, refresh_clicks, stored_data):
        """Update the main data table display"""
        if not stored_data:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-table fa-3x mb-3", style={'color': '#6c757d'}),
                    html.H5("No Data Available", className="text-muted"),
                    html.P("Upload CSV files to see data here", className="text-muted")
                ], className="text-center py-5")
            ])

        try:
            if isinstance(stored_data, dict) and 'data' in stored_data:
                df = pd.DataFrame(stored_data['data'])
            else:
                df = pd.DataFrame(stored_data)
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            return html.Div("Error loading data", className="text-danger text-center p-4")

        is_combined = '_source_file' in df.columns

        if selected_columns:
            if is_combined and '_source_file' not in selected_columns:
                selected_columns = selected_columns + ['_source_file']
            try:
                df_display = df[selected_columns]
            except KeyError:
                df_display = df
        else:
            df_display = df

        if row_limit:
            df_display = df_display.head(row_limit)

        status_badges = [
            html.Span(f"📊 {len(df)} total rows", className="badge bg-primary me-2"),
            html.Span(f"📋 {len(df.columns)-1 if is_combined else len(df.columns)} data columns", className="badge bg-info me-2")
        ]

        if is_combined:
            source_files = df['_source_file'].unique()
            status_badges.append(html.Span(f"📁 {len(source_files)} source files", className="badge bg-success me-2"))
            file_breakdown = html.Div([
                html.H6("Source File Breakdown:", className="mt-3 mb-2"),
                html.Ul([html.Li(f"{file}: {len(df[df['_source_file']==file])} rows") for file in source_files])
            ])
        else:
            file_breakdown = html.Div()

        table = dash_table.DataTable(
            data=df_display.to_dict('records'),
            columns=[{"name": col, "id": col, "type": "numeric" if pd.api.types.is_numeric_dtype(df_display[col]) else "text"}
                    for col in df_display.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '600px', 'border': '1px solid #dee2e6'},
            style_cell={'textAlign': 'left', 'padding': '8px 12px', 'fontFamily': 'Arial, sans-serif',
                       'fontSize': '13px', 'border': '1px solid #dee2e6', 'backgroundColor': '#ffffff'},
            style_header={'backgroundColor': '#4472C4', 'color': 'white', 'fontWeight': 'bold',
                         'textAlign': 'center', 'border': '1px solid #2F5597'},
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#F2F2F2'},
                {'if': {'state': 'selected'}, 'backgroundColor': '#D6EBFF', 'border': '1px solid #0078D4'}
            ],
            sort_action="native", filter_action="native", page_action="native",
            page_current=0, page_size=50, export_format="xlsx",
            export_headers="display", row_selectable="multi"
        )

        return html.Div([html.Div(status_badges, className="mb-3"), file_breakdown, table])

    # TAB SWITCHING - FIXED
    @callback(
    [Output("table-tab-btn", "color"),
     Output("plots-tab-btn", "color"),
     Output("options-tab-btn", "color"),
     Output("table-panel", "style"),
     Output("plots-panel", "style"),
     Output("options-panel", "style")],
    [Input("table-tab-btn", "n_clicks"),
     Input("plots-tab-btn", "n_clicks"),
     Input("options-tab-btn", "n_clicks")],
    prevent_initial_call=True
)
    def switch_tabs(table_clicks, plots_clicks, options_clicks):
        """Switch between CSV tabs only - doesn't affect FITS/images"""
    
        # Initialize with Table tab active (default)
        colors = ["primary", "outline-primary", "outline-secondary"]
        styles = [{"display": "block"}, {"display": "none"}, {"display": "none"}]
    
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
            if button_id == "plots-tab-btn":
                # Charts & Plots tab active
                colors = ["outline-primary", "primary", "outline-secondary"]
                styles = [{"display": "none"}, {"display": "block"}, {"display": "none"}]
            elif button_id == "options-tab-btn":
                # Data Options tab active
                colors = ["outline-primary", "outline-secondary", "primary"]
                styles = [{"display": "none"}, {"display": "none"}, {"display": "block"}]
            # else: Table tab remains active (default colors/styles)
    
        return colors + styles


    @callback(
        Output("data-summary", "children"),
        Input("processed-data", "data")
    )
    def update_data_summary(processed_data):
        """Update data summary"""
        if not processed_data or not processed_data.get('data'):
            return "No CSV data available for summary"

        try:
            if isinstance(processed_data, dict) and 'data' in processed_data:
                df = pd.DataFrame(processed_data['data'])
            else:
                df = pd.DataFrame(processed_data)

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            text_cols = df.select_dtypes(include=['object']).columns

            return html.Div([
                html.P([html.Strong("Dataset Size: "), f"{len(df):,} rows × {len(df.columns)} columns"]),
                html.P([html.Strong("Numeric Columns: "), f"{len(numeric_cols)}"]),
                html.P([html.Strong("Text Columns: "), f"{len(text_cols)}"]),
                html.P([html.Strong("Memory Usage: "), f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"]),
                html.P([html.Strong("Missing Values: "), f"{df.isnull().sum().sum():,}"]),
                html.P([html.Strong("Duplicate Rows: "), f"{df.duplicated().sum():,}"]),
            ])

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    # Default axis selections - FIXED
    @callback(
        [Output("x-data-selector", "value"),
         Output("y-data-selector", "value")],
        [Input("x-data-selector", "options"),
         Input("y-data-selector", "options")],
        prevent_initial_call=True
    )
    def set_default_axis_selections(x_options, y_options):
        """Set smart defaults for axis selections"""
        if x_options and y_options:
            x_default = x_options[0]['value'] if len(x_options) > 0 else None
            y_default = x_options[1]['value'] if len(x_options) > 1 else x_options['value'] if len(x_options) > 0 else None
            return x_default, y_default
        return None, None

    print("✅ All visualization callbacks registered successfully")
    logger.info("✅ All visualization callbacks registered successfully")
