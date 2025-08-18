"""
AstroAid Dashboard - Main Application
A robust file viewer for CSV and FITS files with AI/ML integration capabilities
"""

import dash
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
import config
from src.components.file_upload import create_upload_area
from src.components.data_viewer import create_enhanced_data_viewer
from src.callbacks import file_handling

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="AstroAid Dashboard"
)

# Server instance for deployment
server = app.server

# Main application layout
app.layout = dbc.Container([
    # Data stores for session management
    dcc.Store(id='session-data', storage_type='session'),
    dcc.Store(id='file-metadata', storage_type='session'),
    dcc.Store(id='processed-data', storage_type='memory'),
    dcc.Store(id='fits-data-store', storage_type='session'),  # NEW: Store FITS data

    # Header Section
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-telescope me-3", style={'color': '#0d6efd'}),
                "🌟 AstroAid Dashboard"
            ], className="text-center mb-2", style={'color': '#212529', 'fontWeight': 'bold'}),
            html.P("Professional FITS and CSV file analyzer for astronomical data",
                   className="text-center text-muted mb-4",
                   style={'fontSize': '16px'}),
        ])
    ], className="mb-4"),

    # File Upload Section
    dbc.Row([
        dbc.Col([
            create_upload_area(),
        ], width=12)
    ], className="mb-4"),

    # Clear Button and Upload Status
    html.Div([
        dbc.Button([
            html.I(className="fas fa-broom me-2"),
            "Clear All Uploads"
        ], id="clear-uploads-btn", color="danger", size="sm", className="mb-3"),
        html.Div(id='upload-status')
    ], className="mb-3"),

    # Welcome screen (shown when no data is uploaded)
    html.Div(id='welcome-screen', children=[
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-cloud-upload-alt fa-4x mb-4",
                           style={'color': '#6c757d'}),
                    html.H4("Upload Files to Begin Analysis",
                           className="text-muted mb-3",
                           style={'fontWeight': '500'}),
                    html.P("Drag & drop CSV or FITS files above to start exploring your data",
                           className="text-muted mb-4",
                           style={'fontSize': '14px'}),

                    # Feature highlights
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-table fa-2x mb-2",
                                       style={'color': '#28a745'}),
                                html.H6("Excel-like Tables", className="text-success"),
                                html.P("Interactive data tables with sorting and filtering",
                                       className="small text-muted")
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-chart-line fa-2x mb-2",
                                       style={'color': '#007bff'}),
                                html.H6("Combined Data Plots", className="text-primary"),
                                html.P("Create plots from multiple CSV files combined",
                                       className="small text-muted")
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-satellite fa-2x mb-2",
                                       style={'color': '#6f42c1'}),
                                html.H6("Astronomical Data", className="text-purple"),
                                html.P("Specialized tools for FITS files and astronomy",
                                       className="small text-muted")
                            ], className="text-center")
                        ], width=4)
                    ], className="mt-4")
                ], className="text-center py-5")
            ])
        ], style={'backgroundColor': '#f8f9fa', 'border': '2px dashed #dee2e6'})
    ]),

    # MAIN DISPLAY AREA - Shows FITS images or CSV data
    html.Div(id='main-display-area', children=[]),

    # Footer Section
    html.Footer([
        html.Hr(style={'margin': '2rem 0 1rem 0'}),
        dbc.Row([
            dbc.Col([
                html.P([
                    "AstroAid Dashboard v1.0 - Built with ",
                    html.A("Dash", href="https://dash.plotly.com", target="_blank",
                           className="text-decoration-none"),
                    " | ",
                    html.I(className="fas fa-heart", style={'color': '#e74c3c'}),
                    " Made for Astronomy"
                ], className="text-center text-muted mb-0", style={'fontSize': '13px'})
            ])
        ])
    ], className="mt-5")

], fluid=True, className="px-4", style={'minHeight': '100vh', 'backgroundColor': '#ffffff'})

# Register all callbacks
def register_all_callbacks():
    """Register all application callbacks"""
    try:
        # Register file handling callbacks
        file_handling.register_callbacks(app)
        print("✅ File handling callbacks registered")

        # Register visualization callbacks
        try:
            from src.callbacks import visualization
            visualization.register_callbacks(app)
            print("✅ Visualization callbacks registered")
        except ImportError as e:
            print(f"⚠️ Visualization callbacks not found: {e}")
            print(" Create src/callbacks/visualization.py to enable plotting features")

    except Exception as e:
        print(f"❌ Error registering callbacks: {e}")

# Register callbacks after app definition
register_all_callbacks()

# Development server configuration
if __name__ == '__main__':
    print("🚀 Starting AstroAid Dashboard...")
    print(f"📊 Dash version: {dash.__version__}")
    print(f"🌐 Server URL: http://{config.HOST}:{config.PORT}")
    print("📁 Ready to process CSV and FITS files!")
    print("✨ Excel-style interface with combined CSV plotting")
    print("-" * 60)

    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT,
        dev_tools_ui=True,
        dev_tools_hot_reload=True
    )
