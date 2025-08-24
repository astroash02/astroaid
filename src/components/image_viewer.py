"""
Enhanced image viewer with interactive filters and colormaps for Target 2
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

def create_image_display(image_files):
    """Create comprehensive image display with advanced processing controls"""
    if not image_files:
        return html.Div([
            dbc.Alert("No image files to display", color="warning")
        ])
    
    content = []
    
    # Enhanced Header
    content.append(
        dbc.Card([
            dbc.CardBody([
                html.H3([
                    html.I(className="fas fa-microscope me-2"),
                    f"Advanced Image Analysis - {len(image_files)} file(s)"
                ]),
                html.Div([
                    dbc.Badge(f"{len(image_files)} images", color="info", className="me-2"),
                    dbc.Badge("Interactive Processing", color="success", className="me-2"),
                ])
            ])
        ], className="mb-4")
    )
    
    # Process each image file with advanced controls
    for img_idx, image_file in enumerate(image_files):
        enhanced_card = create_advanced_image_card(image_file, img_idx)
        content.append(enhanced_card)
    
    return html.Div(content)

def create_advanced_image_card(image_file, img_idx):
    """Create advanced image card with interactive filters and colormaps"""
    filename = image_file.get('filename', f'Image_{img_idx}')
    image_data = image_file.get('image_data', {})
    metadata = image_file.get('metadata', {})
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-microscope me-2"),
                filename
            ], className="mb-0"),
            html.Small(
                f"{metadata.get('format', 'Unknown')} • {metadata.get('size', 'Unknown size')}", 
                className="text-muted"
            )
        ]),
        dbc.CardBody([
            dbc.Row([
                # Enhanced Image Display Panel
                dbc.Col([
                    html.Div([
                        html.Img(
                            id={'type': 'processed-image', 'index': img_idx},
                            src=f"data:image/png;base64,{image_data.get('original_image_b64', '')}",
                            style={
                                'width': '100%',
                                'maxHeight': '600px',
                                'objectFit': 'contain',
                                'border': '2px solid #dee2e6',
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                            }
                        ),
                        # Processing status indicator
                        html.Div(
                            id={'type': 'processing-status', 'index': img_idx},
                            className="mt-2 text-center",
                            children=[
                                dbc.Badge("Original", color="secondary", className="me-2"),
                                html.Small("Ready for processing", className="text-muted")
                            ]
                        )
                    ])
                ], width=8),
                
                # Advanced Controls Panel
                dbc.Col([
                    # Filter Controls Section
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="fas fa-filter me-2"),
                                "Image Filters"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Label("Filter Type:", className="form-label fw-bold"),
                            dcc.Dropdown(
                                id={'type': 'filter-type', 'index': img_idx},
                                options=[
                                    {'label': '🚫 None', 'value': 'none'},
                                    {'label': '🌀 Gaussian Blur', 'value': 'gaussian'},
                                    {'label': '🔹 Median Filter', 'value': 'median'},
                                    {'label': '🔍 Sobel Edge Detection', 'value': 'sobel'},
                                    {'label': '✨ Unsharp Mask', 'value': 'unsharp'},
                                    {'label': '☀️ Brightness Adjust', 'value': 'brightness'},
                                    {'label': '🎚️ Contrast Enhance', 'value': 'contrast'}
                                ],
                                value='none',
                                className="mb-3"
                            ),
                            
                            # Filter Parameters
                            html.Label("Filter Strength:", className="form-label fw-bold"),
                            dcc.Slider(
                                id={'type': 'filter-strength', 'index': img_idx},
                                min=0.1, max=3.0, value=1.0, step=0.1,
                                marks={
                                    0.5: {'label': '0.5', 'style': {'fontSize': '10px'}},
                                    1.0: {'label': '1.0', 'style': {'fontSize': '10px'}},
                                    1.5: {'label': '1.5', 'style': {'fontSize': '10px'}},
                                    2.0: {'label': '2.0', 'style': {'fontSize': '10px'}},
                                    2.5: {'label': '2.5', 'style': {'fontSize': '10px'}}
                                },
                                className="mb-3",
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Colormap Controls Section
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="fas fa-palette me-2"),
                                "Colormaps"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Label("Colormap:", className="form-label fw-bold"),
                            dcc.Dropdown(
                                id={'type': 'colormap', 'index': img_idx},
                                options=[
                                    {'label': '🎨 Original', 'value': 'original'},
                                    {'label': '⚫ Grayscale', 'value': 'grayscale'},
                                    {'label': '🔥 Hot (Heat)', 'value': 'hot'},
                                    {'label': '❄️ Cool (Blue)', 'value': 'cool'},
                                    {'label': '🌈 Viridis', 'value': 'viridis'},
                                    {'label': '🌟 Plasma', 'value': 'plasma'},
                                    {'label': '🌋 Inferno', 'value': 'inferno'},
                                    {'label': '🔮 Magma', 'value': 'magma'}
                                ],
                                value='original',
                                className="mb-3"
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Action Buttons Section
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="fas fa-cogs me-2"),
                                "Actions"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Button([
                                html.I(className="fas fa-play me-2"),
                                "Apply Processing"
                            ], 
                            id={'type': 'apply-processing', 'index': img_idx}, 
                            color="primary", 
                            size="sm", 
                            className="w-100 mb-2"),
                            
                            dbc.Button([
                                html.I(className="fas fa-undo me-2"),
                                "Reset to Original"
                            ], 
                            id={'type': 'reset-processing', 'index': img_idx}, 
                            color="secondary", 
                            size="sm", 
                            className="w-100 mb-2"),
                            
                            dbc.Button([
                                html.I(className="fas fa-download me-2"),
                                "Export Processed"
                            ], 
                            id={'type': 'export-image', 'index': img_idx}, 
                            color="success", 
                            size="sm", 
                            className="w-100")
                        ])
                    ])
                    
                ], width=4)
            ]),
            
            # Hidden data store for original image
            dcc.Store(
                id={'type': 'original-image-data', 'index': img_idx},
                data={
                    'numpy_data': image_data.get('numpy_data', []),
                    'filename': filename,
                    'original_b64': image_data.get('original_image_b64', '')
                }
            ),
            
            # Image metadata section
            html.Hr(),
            html.Div([
                html.H6("📋 Image Information", className="mb-2"),
                html.P([html.Strong("Format: "), metadata.get('format', 'Unknown')]),
                html.P([html.Strong("Size: "), str(metadata.get('size', 'Unknown'))]),
                html.P([html.Strong("Data Type: "), metadata.get('data_type', 'Unknown')])
            ])
        ])
    ], className="mb-4")
