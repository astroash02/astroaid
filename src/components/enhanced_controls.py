"""
Enhanced UI Controls for WCS-enabled FITS Processing
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

def create_wcs_info_display(wcs_info):
    """Create a compact WCS information display"""
    if not wcs_info or not wcs_info.get('wcs_available'):
        return html.Div([
            dbc.Badge("No WCS", color="secondary", className="me-2"),
            html.Small("Pixel coordinates only", className="text-muted")
        ])
    
    coord_system = wcs_info.get('coordinate_system', ['Unknown', 'Unknown'])
    projection = wcs_info.get('projection', 'UNK')
    pixel_scale = wcs_info.get('pixel_scale', {})
    scale_text = f"{pixel_scale.get('average_arcsec', 0):.2f}\"/px" if pixel_scale else "Unknown scale"
    
    return html.Div([
        dbc.Badge("🌍 WCS Available", color="success", className="me-2"),
        html.Small([
            f"{coord_system[0]}/{coord_system[1]} • {projection} • {scale_text}"
        ], className="text-success")
    ])

def create_enhanced_filter_controls(unique_id):
    """Create enhanced filter controls with WCS awareness"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("🎛️ Filter Type:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id={'type': 'fits-filter', 'index': unique_id},
                    options=[
                        {'label': '🔍 None (Original)', 'value': 'none'},
                        {'label': '🌟 Brightness Boost', 'value': 'brightness'},
                        {'label': '🎯 Contrast Enhancement', 'value': 'contrast'},
                        {'label': '🌀 Gaussian Smooth', 'value': 'gaussian'},
                        {'label': '🔪 Unsharp Mask', 'value': 'sharpen'},
                        {'label': '📊 Histogram Equalization', 'value': 'enhance'},
                        {'label': '🎲 Median Filter', 'value': 'median'}
                    ],
                    value='none',
                    clearable=False,
                    className="mb-2"
                )
            ], width=6),
            
            dbc.Col([
                html.Label("⚡ Filter Strength:", className="form-label fw-bold"),
                dcc.Slider(
                    id={'type': 'fits-strength', 'index': unique_id},
                    min=0.1, max=3.0, value=1.0, step=0.1,
                    marks={0.1: '0.1', 0.5: '0.5', 1: '1', 2: '2', 3: '3'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=6)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Label("🎨 Color Mapping:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id={'type': 'fits-colormap', 'index': unique_id},
                    options=[
                        {'label': '🌈 Viridis (Recommended)', 'value': 'Viridis'},
                        {'label': '🔥 Plasma (High Contrast)', 'value': 'Plasma'},
                        {'label': '🌋 Inferno (Warm)', 'value': 'Inferno'},
                        {'label': '🎨 Magma (Purple)', 'value': 'Magma'},
                        {'label': '🌊 Cividis (Colorblind Safe)', 'value': 'Cividis'},
                        {'label': '🌡️ Hot (Classic)', 'value': 'Hot'},
                        {'label': '🌈 Jet (Rainbow)', 'value': 'Jet'},
                        {'label': '⚫ Greys (Monochrome)', 'value': 'Greys'},
                        {'label': '🔵 Blues (Cool)', 'value': 'Blues'},
                        {'label': '🔴 Reds (Warm)', 'value': 'Reds'}
                    ],
                    value='Viridis',
                    clearable=False,
                    className="mb-2"
                )
            ], width=6),
            
            dbc.Col([
                html.Label("📏 Intensity Scale:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id={'type': 'fits-scale', 'index': unique_id},
                    options=[
                        {'label': '📏 Linear (Default)', 'value': 'linear'},
                        {'label': '📊 Logarithmic (High DR)', 'value': 'log'},
                        {'label': '📐 Square Root (Gentle)', 'value': 'sqrt'},
                        {'label': '🌌 Asinh (Astronomical)', 'value': 'asinh'},
                        {'label': '⚡ Power Law (Custom)', 'value': 'power'}
                    ],
                    value='linear',
                    clearable=False,
                    className="mb-2"
                )
            ], width=6)
        ], className="mb-3")
    ])

def create_coordinate_display_card(wcs_info):
    """Create a card showing coordinate system information"""
    if not wcs_info or not wcs_info.get('wcs_available'):
        return dbc.Card([
            dbc.CardBody([
                html.H6("📍 Coordinate Information", className="card-title"),
                html.P("No world coordinate system available", className="text-muted"),
                html.P("Image coordinates shown in pixels only", className="small text-info")
            ])
        ], className="mb-3")
    
    coord_system = wcs_info.get('coordinate_system', ['Unknown', 'Unknown'])
    projection_name = wcs_info.get('projection_name', 'Unknown')
    pixel_scale = wcs_info.get('pixel_scale', {})
    
    return dbc.Card([
        dbc.CardBody([
            html.H6("🌍 World Coordinate System", className="card-title text-success"),
            html.Div([
                html.Strong("Coordinate System: "),
                f"{coord_system[0]} / {coord_system[1]}"
            ], className="mb-2"),
            html.Div([
                html.Strong("Projection: "),
                projection_name
            ], className="mb-2"),
            html.Div([
                html.Strong("Pixel Scale: "),
                f"{pixel_scale.get('average_arcsec', 0):.3f} arcsec/pixel" if pixel_scale else "Unknown"
            ], className="mb-2"),
            html.Small("Hover over image for RA/Dec coordinates", className="text-info")
        ])
    ], className="mb-3")
