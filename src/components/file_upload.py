"""
File upload component with drag-and-drop support
ENHANCED with IMAGE FORMAT SUPPORT - FIXED OVERLAPS
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

def create_upload_area():
    """Create the file upload area component - FIXED OVERLAPPING BADGES"""
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
                        
                        # FIXED: Data Files row with proper spacing
                        html.Div([
                            html.Div([
                                html.Span("Data Files: ", className="text-muted small me-2"),
                                dbc.Badge("CSV", color="primary", className="me-1", style={'fontSize': '9px'}),
                                dbc.Badge("FITS", color="info", className="me-1", style={'fontSize': '9px'}),
                                dbc.Badge("TSV", color="success", className="me-1", style={'fontSize': '9px'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexWrap': 'wrap'})
                        ], className="mb-2"),
                        
                        # FIXED: Images row with proper spacing
                        html.Div([
                            html.Div([
                                html.Span("Images: ", className="text-muted small me-2"),
                                dbc.Badge("JPG", color="warning", className="me-1", style={'fontSize': '9px'}),
                                dbc.Badge("PNG", color="info", className="me-1", style={'fontSize': '9px'}),
                                dbc.Badge("TIFF", color="success", className="me-1", style={'fontSize': '9px'}),
                                dbc.Badge("BMP", color="secondary", className="me-1", style={'fontSize': '9px'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexWrap': 'wrap'})
                        ], className="mb-2"),
                        
                        # FIXED: Scientific row with proper spacing
                        html.Div([
                            html.Div([
                                html.Span("Scientific: ", className="text-muted small me-2"),
                                dbc.Badge("NPY", color="dark", className="me-1", style={'fontSize': '9px'}),
                                dbc.Badge("NPZ", color="dark", className="me-1", style={'fontSize': '9px'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexWrap': 'wrap'})
                        ], className="mb-3"),
                        
                        # File size limit
                        html.P("Max size: 500MB per file",
                               className="text-muted small mb-0",
                               style={'fontSize': '11px'})
                    ], className="text-center", style={
                        'lineHeight': 'normal', 
                        'padding': '1.2rem',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    })
                ]),
                style={
                    'width': '100%',
                    'height': '180px',  # INCREASED height to accommodate all badges properly
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '8px',
                    'borderColor': '#007bff',
                    'textAlign': 'center',
                    'backgroundColor': '#f8f9ff',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'maxWidth': '550px',  
                    'margin': '0 auto',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'overflow': 'hidden'  # PREVENT any content overflow
                },
                multiple=True,
                accept='.csv,.tsv,.txt,.dat,.fits,.fit,.fts,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.gif,.npy,.npz'
            ),
            
            # Upload progress (initially hidden)
            html.Div(id='upload-progress', style={'display': 'none'})
        ], style={'padding': '1.5rem'})
    ],
    className="mb-6",
    style={
        'maxWidth': '580px',  # INCREASED container width
        'margin': '0 auto',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })
