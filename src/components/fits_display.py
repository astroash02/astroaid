"""
Enhanced FITS Display Components with WCS Support
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import logging
import numpy as np
logger = logging.getLogger(__name__)

def create_enhanced_fits_display(fits_files):
    """Create comprehensive FITS display with WCS information"""
    if not fits_files:
        return html.Div([
            dbc.Alert("No FITS files to display", color="warning")
        ])

    content = []
    
    # Enhanced header with WCS summary
    total_hdus = sum(len(f.get('hdu_list', [])) for f in fits_files)
    wcs_hdus = sum(1 for f in fits_files for hdu in f.get('hdu_list', []) 
                   if hdu.get('wcs_info', {}).get('wcs_available', False))
    
    content.append(
        dbc.Card([
            dbc.CardBody([
                html.H3([
                    html.I(className="fas fa-satellite me-2"),
                    f"Enhanced FITS Analysis - {len(fits_files)} file(s), {total_hdus} HDUs"
                ]),
                html.Div([
                    dbc.Badge(f"{len(fits_files)} FITS files", color="info", className="me-2"),
                    dbc.Badge(f"{total_hdus} HDUs total", color="success", className="me-2"),
                    dbc.Badge(f"{wcs_hdus} with WCS", color="primary", className="me-2"),
                    dbc.Badge("🌍 World Coordinates", color="warning", className="me-2") if wcs_hdus > 0 else None,
                ])
            ])
        ], className="mb-4")
    )

    # Process each FITS file with enhanced WCS display
    for file_idx, fits_file in enumerate(fits_files):
        filename = fits_file.get('filename', f'File_{file_idx}')
        hdu_list = fits_file.get('hdu_list', [])
        
        if not hdu_list:
            continue

        # Enhanced file header with WCS info
        file_wcs_count = sum(1 for hdu in hdu_list if hdu.get('wcs_info', {}).get('wcs_available', False))
        
        content.append(
            html.H4([
                html.I(className="fas fa-file me-2"),
                filename,
                html.Small(f" ({file_wcs_count} HDUs with WCS)" if file_wcs_count > 0 else " (No WCS)", 
                          className="text-muted ms-2")
            ], className="mt-4 mb-3")
        )

        # Process each HDU with enhanced WCS cards
        hdu_rows = []
        for hdu_idx in range(0, len(hdu_list), 2):  # Two HDUs per row
            hdu_cols = []
            for i in range(2):
                if hdu_idx + i < len(hdu_list):
                    hdu = hdu_list[hdu_idx + i]
                    hdu_card = create_enhanced_hdu_card(hdu, filename)
                    hdu_cols.append(dbc.Col([hdu_card], width=6, className="mb-3"))
            
            if hdu_cols:
                hdu_rows.append(dbc.Row(hdu_cols))
        
        content.extend(hdu_rows)

    return html.Div(content)

def create_enhanced_hdu_card(hdu, filename):
    """Create enhanced HDU card with comprehensive WCS information"""
    hdu_type = hdu.get('type', 'Unknown')
    hdu_index = hdu.get('index', 0)
    hdu_info = hdu.get('info', '')
    wcs_info = hdu.get('wcs_info', {})
    
    # Enhanced card header with WCS indicator
    has_wcs = wcs_info.get('wcs_available', False)
    wcs_indicator = html.I(className="fas fa-globe-americas text-success ms-2", 
                          title="World Coordinates Available") if has_wcs else None
    
    card_header = dbc.CardHeader([
        html.H6([
            html.I(className=f"fas fa-{'image' if hdu_type == 'IMAGE' else 'table' if hdu_type == 'TABLE' else 'file-alt'} me-2"),
            f"HDU {hdu_index}: {hdu_type}",
            wcs_indicator
        ], className="mb-0"),
        html.Small(hdu_info, className="text-muted")
    ])

    # Enhanced card body content
    card_body_content = []

    # Display image if available
    if hdu.get('image_b64'):
        card_body_content.append(
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{hdu['image_b64']}",
                    style={
                        'width': '100%',
                        'maxHeight': '400px',
                        'objectFit': 'contain',
                        'border': '1px solid #dee2e6',
                        'borderRadius': '4px'
                    }
                )
            ], className="mb-3")
        )

    # Enhanced WCS information display
    if has_wcs:
        wcs_items = []
        
        # Coordinate system
        coord_system = wcs_info.get('coordinate_system', ['Unknown', 'Unknown'])
        wcs_items.append(
            html.Tr([
                html.Td(html.Code("Coordinate System"), style={'width': '40%'}),
                html.Td(f"{coord_system[0]} / {coord_system[1]}")
            ])
        )
        
        # Pixel scale
        pixel_scale = wcs_info.get('pixel_scale')
        if pixel_scale:
            wcs_items.append(
                html.Tr([
                    html.Td(html.Code("Pixel Scale")),
                    html.Td(f"{pixel_scale.get('average_arcsec', 0):.3f} arcsec/pixel")
                ])
            )
            
            # Field of view (if we can calculate it)
            try:
                # This would require image dimensions from raw data
                # For now, just show pixel scale details
                wcs_items.append(
                    html.Tr([
                        html.Td(html.Code("Scale Detail")),
                        html.Td(f"X: {pixel_scale.get('x_scale_arcsec', 0):.3f}\", Y: {pixel_scale.get('y_scale_arcsec', 0):.3f}\"")
                    ])
                )
            except:
                pass
        
        # Projection
        projection_name = wcs_info.get('projection_name', 'Unknown')
        projection_code = wcs_info.get('projection', 'UNK')
        wcs_items.append(
            html.Tr([
                html.Td(html.Code("Projection")),
                html.Td(f"{projection_name} ({projection_code})")
            ])
        )
        
        # Reference coordinates
        ref_coord = wcs_info.get('reference_coordinate')
        if ref_coord and len(ref_coord) >= 2:
            wcs_items.append(
                html.Tr([
                    html.Td(html.Code("Reference Point")),
                    html.Td(f"RA: {ref_coord[0]:.6f}°, Dec: {ref_coord[1]:.6f}°")
                ])
            )
        
        # Coordinate frame
        coordinate_details = wcs_info.get('coordinate_details', {})
        coord_frame = coordinate_details.get('coordinate_frame', 'Unknown')
        equinox = coordinate_details.get('equinox', 'Unknown')
        if coord_frame != 'Unknown':
            wcs_items.append(
                html.Tr([
                    html.Td(html.Code("Reference Frame")),
                    html.Td(f"{coord_frame} (J{equinox})" if equinox != 'Unknown' else coord_frame)
                ])
            )
        
        wcs_table = dbc.Table([
            html.Tbody(wcs_items)
        ], size="sm", className="mb-0")
        
        card_body_content.extend([
            html.H6("🌍 World Coordinate System:", className="mt-3 mb-2"),
            wcs_table
        ])
    else:
        reason = wcs_info.get('reason', 'No WCS headers found')
        card_body_content.extend([
            html.H6("🌍 World Coordinates:", className="mt-3 mb-2"),
            html.P([
                html.I(className="fas fa-info-circle me-2"),
                f"Not available: {reason}"
            ], className="text-muted small")
        ])

    # Interactive processing section with WCS data
    if hdu_type == 'IMAGE' and hdu.get('raw_image_data'):
        unique_id = f"{filename}_{hdu_index}".replace(' ', '_').replace('.', '_')
        
        card_body_content.extend([
            html.Hr(),
            html.H6("🔧 Interactive Processing:", className="mb-3"),
            
            # Enhanced controls row
            dbc.Row([
                dbc.Col([
                    html.Label("Filter:", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id={'type': 'fits-filter', 'index': unique_id},
                        options=[
                            {'label': '🔍 None', 'value': 'none'},
                            {'label': '🌟 Brightness', 'value': 'brightness'},
                            {'label': '🎯 Contrast', 'value': 'contrast'},
                            {'label': '🌀 Gaussian Smooth', 'value': 'gaussian'},
                            {'label': '🔪 Sharpen', 'value': 'sharpen'},
                            {'label': '📊 Enhance', 'value': 'enhance'},
                            {'label': '🎲 Median', 'value': 'median'}
                        ],
                        value='none',
                        clearable=False
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Strength:", className="form-label fw-bold"),
                    dcc.Slider(
                        id={'type': 'fits-strength', 'index': unique_id},
                        min=0.1, max=3.0, value=1.0, step=0.1,
                        marks={0.5: '0.5', 1: '1', 2: '2', 3: '3'}
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Colormap:", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id={'type': 'fits-colormap', 'index': unique_id},
                        options=[
                            {'label': '🌈 Viridis (Recommended)', 'value': 'Viridis'},
                            {'label': '🔥 Plasma', 'value': 'Plasma'},
                            {'label': '🌋 Inferno', 'value': 'Inferno'},
                            {'label': '🎨 Magma', 'value': 'Magma'},
                            {'label': '🌊 Cividis', 'value': 'Cividis'},
                            {'label': '🌡️ Hot', 'value': 'Hot'},
                            {'label': '🌈 Jet', 'value': 'Jet'},
                            {'label': '⚫ Greys', 'value': 'Greys'}
                        ],
                        value='Viridis',
                        clearable=False
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Intensity Scale:", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id={'type': 'fits-scale', 'index': unique_id},
                        options=[
                            {'label': '📏 Linear', 'value': 'linear'},
                            {'label': '📊 Logarithmic', 'value': 'log'},
                            {'label': '📐 Square Root', 'value': 'sqrt'},
                            {'label': '🌌 Asinh (Astro)', 'value': 'asinh'},
                            {'label': '⚡ Power Law', 'value': 'power'}
                        ],
                        value='linear',
                        clearable=False
                    )
                ], width=6)
            ], className="mb-3"),
            
            # Action buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-magic me-2"),
                        "Apply Processing"
                    ], id={'type': 'fits-apply', 'index': unique_id}, 
                    color="primary", size="sm", className="w-100")
                ], width=6),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-undo me-2"),
                        "Reset Original"
                    ], id={'type': 'fits-reset', 'index': unique_id}, 
                    color="secondary", size="sm", className="w-100")
                ], width=6)
            ], className="mb-3"),
            
            # Interactive plot with WCS support
            dcc.Graph(
                id={'type': 'fits-interactive-plot', 'index': unique_id},
                figure=create_initial_fits_figure(hdu.get('raw_image_data'), f"{filename} HDU {hdu_index}", wcs_info),
                config={'displayModeBar': True, 'displaylogo': False}
            ),
            
            # Status display
            html.Div(id={'type': 'fits-region-info', 'index': unique_id}, className="mt-2"),
            
            # Hidden data stores
            dcc.Store(id={'type': 'fits-original-data', 'index': unique_id}, 
                     data={'image_data': hdu.get('raw_image_data')}),
            dcc.Store(id={'type': 'fits-wcs-info', 'index': unique_id}, 
                     data=wcs_info)  # NEW: Store WCS information
        ])

    # Display table data if available
    table_data = hdu.get('table_data')
    if table_data is not None and isinstance(table_data, dict) and 'data' in table_data:
        try:
            from dash import dash_table
            preview_data = table_data['data'][:5] if len(table_data['data']) > 5 else table_data['data']
            columns = table_data.get('columns', [])
            preview_columns = [{"name": col, "id": col} for col in columns[:6]]
            
            card_body_content.extend([
                html.Hr(),
                html.H6("📊 Table Data Preview:", className="mt-3 mb-2"),
                html.P(f"{len(table_data['data'])} rows × {len(columns)} columns", className="small text-muted mb-2"),
                dash_table.DataTable(
                    data=preview_data,
                    columns=preview_columns,
                    style_table={'overflowX': 'auto', 'maxHeight': '200px'},
                    style_cell={'textAlign': 'left', 'fontSize': '11px', 'padding': '4px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    page_size=5
                )
            ])
        except Exception as table_error:
            logger.debug(f"Error processing table data: {table_error}")

    # Display enhanced header information including WCS keywords
    if hdu.get('header') and isinstance(hdu['header'], dict):
        header_info = hdu['header']
        # Enhanced header keys including WCS
        important_keys = ['OBJECT', 'DATE-OBS', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME', 
                         'NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 
                         'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'EQUINOX', 'RADESYS']
        important_header = {k: v for k, v in header_info.items() if k in important_keys and v}
        
        if important_header:
            header_items = []
            for key, value in list(important_header.items())[:8]:  # Show more for WCS
                header_items.append(
                    html.Tr([
                        html.Td(html.Code(key), style={'width': '30%'}),
                        html.Td(str(value)[:60] + ('...' if len(str(value)) > 60 else ''))
                    ])
                )
            
            header_table = dbc.Table([
                html.Tbody(header_items)
            ], size="sm", className="mb-0")
            
            card_body_content.extend([
                html.Hr(),
                html.H6("📋 FITS Header Keywords:", className="mt-3 mb-2"),
                header_table
            ])

    # If no content, show message
    if not card_body_content:
        card_body_content = [
            html.P("No data available for this HDU", className="text-muted text-center py-3")
        ]

    return dbc.Card([
        card_header,
        dbc.CardBody(card_body_content)
    ])

def create_initial_fits_figure(image_data, title, wcs_info):
    """Create initial FITS figure with WCS information"""
    if not image_data:
        return {'data': [], 'layout': {'title': 'No data available'}}
    
    data_array = np.array(image_data)
    
    # Determine labels based on WCS
    has_wcs = wcs_info.get('wcs_available', False) if wcs_info else False
    
    if has_wcs:
        coord_system = wcs_info.get('coordinate_system', ['RA', 'Dec'])
        x_title = f"{coord_system[0]} (pixels)"
        y_title = f"{coord_system[1]} (pixels)"
    else:
        x_title = 'X (pixels)'
        y_title = 'Y (pixels)'
    
    return {
        'data': [{
            'type': 'heatmap',
            'z': data_array.tolist(),
            'colorscale': 'Viridis',
            'showscale': True,
            'hovertemplate': f'X: %{{x}}<br>Y: %{{y}}<br>Intensity: %{{z:.2f}}<extra></extra>'
        }],
        'layout': {
            'title': title,
            'xaxis': {'title': x_title, 'showgrid': False, 'zeroline': False},
            'yaxis': {'title': y_title, 'showgrid': False, 'zeroline': False, 'autorange': 'reversed'},
            'dragmode': 'zoom',
            'height': 400,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    }
