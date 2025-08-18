"""
File upload component with drag-and-drop support
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

def create_upload_area():
    """Create the file upload area component - IMPROVED SPACING"""
    return dbc.Card([
        dbc.CardBody([
            dcc.Upload(
                id='file-upload',
                children=html.Div([
                    html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-2x mb-3", 
                              style={'color': '#007bff'}),
                        html.H6("Drag & Drop Files Here", 
                               className="text-primary mb-3",
                               style={'fontWeight': '600'}),
                        html.P("or click to browse", 
                              className="text-muted mb-3",
                              style={'fontSize': '13px'}),
                        
                        # Supported formats - IMPROVED SPACING
                        html.Div([
                            html.Span("Supported: ", 
                                     className="text-muted small me-2"),
                            dbc.Badge("CSV", color="primary", className="me-2", style={'fontSize': '10px'}),
                            dbc.Badge("FITS", color="info", className="me-2", style={'fontSize': '10px'}),
                            dbc.Badge("TSV", color="success", className="me-2", style={'fontSize': '10px'}),
                        ], className="mb-3"),  # INCREASED spacing
                        
                        # File size limit
                        html.P("Max size: 500MB", 
                              className="text-muted small mb-0",
                              style={'fontSize': '11px'})
                    ], className="text-center", style={'lineHeight': 'normal', 'padding': '1rem'})
                ]),
                style={
                    'width': '100%',
                    'height': '140px',  # INCREASED from 120px for better spacing
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '8px',
                    'borderColor': '#007bff',
                    'textAlign': 'center',
                    'backgroundColor': '#f8f9ff',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'maxWidth': '450px',
                    'margin': '0 auto',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                },
                multiple=True,
                accept='.csv,.tsv,.txt,.dat,.fits,.fit,.fts'
            ),
            
            # Upload progress (initially hidden)
            html.Div(id='upload-progress', style={'display': 'none'})
        ], style={'padding': '1.5rem'})  # ADDED card body padding
    ], 
    className="mb-6",  # INCREASED from mb-4 to mb-6
    style={
        'maxWidth': '500px',
        'margin': '0 auto',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'  # ADDED subtle shadow
    })