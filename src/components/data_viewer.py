from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def create_enhanced_data_viewer():
    """Create Excel-style data viewer with tabs - SINGLE INSTANCE ONLY"""
    return html.Div([
        # Excel-style tab buttons - ONLY ONE SET
        html.Div([
            dbc.ButtonGroup([
                dbc.Button([
                    html.I(className="fas fa-table me-2"),
                    "Table View"
                ],
                id="table-tab-btn",
                color="primary",
                className="px-4 py-2",
                style={"fontWeight": "500", "fontSize": "14px"}),
                dbc.Button([
                    html.I(className="fas fa-chart-line me-2"),
                    "Charts & Plots"
                ],
                id="plots-tab-btn",
                color="outline-primary",
                className="px-4 py-2",
                style={"fontWeight": "500", "fontSize": "14px"}),
                dbc.Button([
                    html.I(className="fas fa-cog me-2"),
                    "Data Options"
                ],
                id="options-tab-btn",
                color="outline-secondary",
                className="px-4 py-2",
                style={"fontWeight": "500", "fontSize": "14px"})
            ], className="mb-4")
        ], className="d-flex justify-content-center"),

        # Table View Panel - ONLY ONE SET OF CONTROLS
        html.Div(id="table-panel", style={"display": "block"}, children=[
            # Quick table controls
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Rows:", className="form-label me-2",
                                           style={"fontSize": "12px", "fontWeight": "500"}),
                                dcc.Input(id="quick-row-limit", type="number", value=100,
                                          min=1, max=10000, className="form-control form-control-sm",
                                          style={"width": "80px", "display": "inline-block"})
                            ], className="d-flex align-items-center me-3"),
                            html.Div([
                                html.Label("Columns:", className="form-label me-2",
                                           style={"fontSize": "12px", "fontWeight": "500"}),
                                dcc.Dropdown(
                                    id="quick-column-selector",
                                    multi=True,
                                    placeholder="All columns",
                                    className="form-control-sm",
                                    style={"minWidth": "200px", "display": "inline-block"}
                                )
                            ], className="d-flex align-items-center me-3"),
                            dbc.Button([
                                html.I(className="fas fa-download me-1"),
                                "Export"
                            ], id="export-btn", color="success", size="sm", className="me-2"),
                            dbc.Button([
                                html.I(className="fas fa-sync me-1"),
                                "Refresh"
                            ], id="refresh-btn", color="secondary", size="sm")
                        ], className="d-flex align-items-center flex-wrap")
                    ])
                ])
            ], className="mb-3"),
            html.Div(id="main-data-table")
        ]),

        # Plots Panel - ONLY ONE SET
        html.Div(id="plots-panel", style={"display": "none"}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Chart Builder"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            # Chart type selection
                            html.Div([
                                html.Label("Chart Type:", className="form-label"),
                                dcc.Dropdown(
                                    id="chart-type-selector",
                                    options=[
                                        {"label": "📊 Column Chart", "value": "bar"},
                                        {"label": "📈 Line Chart", "value": "line"},
                                        {"label": "🔵 Scatter Plot", "value": "scatter"},
                                        {"label": "📦 Box Plot", "value": "box"},
                                        {"label": "📊 Histogram", "value": "histogram"}
                                    ],
                                    value="scatter",
                                    className="mb-3"
                                )
                            ]),
                            # Data selection
                            html.Div([
                                html.Label("X-Axis Data:", className="form-label"),
                                dcc.Dropdown(id="x-data-selector", className="mb-3")
                            ]),
                            html.Div([
                                html.Label("Y-Axis Data:", className="form-label"),
                                dcc.Dropdown(id="y-data-selector", className="mb-3")
                            ]),
                            # Optional styling
                            html.Div([
                                html.Label("Color By:", className="form-label"),
                                dcc.Dropdown(id="color-by-selector", placeholder="None", className="mb-3")
                            ]),
                            html.Div([
                                html.Label("Size By:", className="form-label"),
                                dcc.Dropdown(id="size-by-selector", placeholder="None", className="mb-3")
                            ]),
                            # Action buttons
                            html.Div([
                                dbc.Button([
                                    html.I(className="fas fa-play me-2"),
                                    "Create Chart"
                                ], id="create-chart-btn", color="primary", className="w-100 mb-2"),
                                dbc.Button([
                                    html.I(className="fas fa-save me-2"),
                                    "Save Chart"
                                ], id="save-chart-btn", color="success", size="sm", className="w-100")
                            ])
                        ])
                    ])
                ], width=3),
                # Chart display area
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id="excel-style-chart",
                                style={"height": "500px"},
                                config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}}
                            )
                        ])
                    ])
                ], width=9)
            ])
        ]),

        # Data Options Panel - ONLY ONE SET
        html.Div(id="options-panel", style={"display": "none"}, children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Data Information", className="mb-3"),
                            html.Div(id="data-summary", className="p-3 bg-light rounded")
                        ], width=6),
                        dbc.Col([
                            html.H6("Export Options", className="mb-3"),
                            html.Div([
                                dbc.Button("📄 Export as CSV", color="info", className="w-100 mb-2"),
                                dbc.Button("📊 Export as Excel", color="success", className="w-100 mb-2"),
                                dbc.Button("🖼️ Export Chart as PNG", color="warning", className="w-100 mb-2")
                            ])
                        ], width=6)
                    ])
                ])
            ])
        ])
    ], style={'padding': '20px'})
