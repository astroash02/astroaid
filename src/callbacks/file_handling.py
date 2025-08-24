"""
Enhanced File Handling with Complete Display - FIXED VERSION WITH WORKING FILTERS
"""

from dash import Input, Output, State, callback, html, no_update, ctx, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
import traceback
import pandas as pd
import base64
import io
import numpy as np
from typing import List, Dict, Any
import gc

# Import processors with error handling
try:
    from src.processors.csv_processor import CSVProcessor
    csv_processor = CSVProcessor()
    print("✅ CSV Processor loaded")
except ImportError as e:
    print(f"❌ CSV Processor failed: {e}")
    csv_processor = None

try:
    from src.processors.fits_processor import FITSProcessor
    fits_processor = FITSProcessor()
    print("✅ FITS Processor loaded")
except ImportError as e:
    print(f"❌ FITS Processor failed: {e}")
    fits_processor = None

try:
    from src.processors.image_processor import ImageProcessor
    image_processor = ImageProcessor()
    print("✅ Image Processor loaded")
except ImportError as e:
    print(f"❌ Image Processor failed: {e}")
    image_processor = None

logger = logging.getLogger(__name__)

def register_callbacks(app):
    """Register file handling callbacks with complete display functionality"""
    
    print("🔧 Registering file upload callback...")
    
    @app.callback(
        [Output('session-data', 'data', allow_duplicate=True),
         Output('file-metadata', 'data', allow_duplicate=True),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('main-display-area', 'children', allow_duplicate=True),
         Output('processed-data', 'data', allow_duplicate=True),
         Output('fits-data-store', 'data', allow_duplicate=True),
         Output('file-upload', 'contents')],
        [Input('file-upload', 'contents')],
        [State('file-upload', 'filename'),
         State('file-upload', 'last_modified'),
         State('session-data', 'data')],
        prevent_initial_call=True
    )
    def handle_file_upload(list_of_contents, list_of_names, list_of_dates, existing_data):
        """Handle file upload with complete display functionality and TIFF support"""
        
        print(f"🔍 UPLOAD CALLBACK TRIGGERED!")
        print(f"🔍 Contents received: {list_of_contents is not None}")
        print(f"🔍 Filenames: {list_of_names}")
        
        if not list_of_contents:
            print("🔍 No contents, preventing update")
            if existing_data:
                raise PreventUpdate
            else:
                return None, None, None, html.Div(), None, None, None
        
        print(f"🔍 Processing {len(list_of_contents)} files...")
        
        try:
            uploaded_files = []
            csv_files = []
            fits_files = []
            image_files = []
            
            # Process each file
            for i, contents in enumerate(list_of_contents):
                filename = list_of_names[i] if list_of_names and i < len(list_of_names) else f"unknown_{i}"
                file_ext = filename.lower().split('.')[-1]
                
                print(f"🔍 Processing file {i+1}: {filename} (ext: {file_ext})")
                logger.info(f"Processing uploaded file {i+1}/{len(list_of_contents)}: {filename}")
                
                # CSV Files
                if file_ext in ['csv', 'tsv', 'dat', 'txt']:
                    print(f"🔍 Detected CSV file: {filename}")
                    if csv_processor is None:
                        print("❌ CSV processor not available")
                        uploaded_files.append({'type': 'csv', 'filename': filename, 'status': 'error', 'error': 'CSV processor not loaded'})
                        continue
                        
                    try:
                        df, metadata = csv_processor.process_file(contents, filename)
                        csv_files.append({'df': df, 'metadata': metadata, 'filename': filename})
                        uploaded_files.append({'type': 'csv', 'filename': filename, 'status': 'success', 'rows': len(df)})
                        print(f"✅ CSV processing successful for {filename}: {df.shape}")
                    except Exception as e:
                        print(f"❌ CSV processing failed for {filename}: {e}")
                        uploaded_files.append({'type': 'csv', 'filename': filename, 'status': 'error', 'error': str(e)})
                
                # FITS Files
                elif file_ext in ['fits', 'fit', 'fts']:
                    print(f"🔍 Detected FITS file: {filename}")
                    if fits_processor is None:
                        print("❌ FITS processor not available")
                        uploaded_files.append({'type': 'fits', 'filename': filename, 'status': 'error', 'error': 'FITS processor not loaded'})
                        continue
                        
                    try:
                        hdu_list, file_metadata = fits_processor.process_file(contents, filename)
                        fits_files.append({'hdu_list': hdu_list, 'metadata': file_metadata, 'filename': filename})
                        uploaded_files.append({'type': 'fits', 'filename': filename, 'status': 'success', 'hdus': len(hdu_list)})
                        print(f"✅ FITS processing successful for {filename}: {len(hdu_list)} HDUs")
                    except Exception as e:
                        print(f"❌ FITS processing failed for {filename}: {e}")
                        uploaded_files.append({'type': 'fits', 'filename': filename, 'status': 'error', 'error': str(e)})
                
                # Image Files with TIFF Support
                elif file_ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'gif', 'npy', 'npz']:
                    print(f"🔍 Detected image file: {filename}")
                    
                    try:
                        # Handle TIFF files specially
                        if file_ext in ['tiff', 'tif']:
                            print(f"🔍 Converting TIFF to PNG for browser compatibility: {filename}")
                            
                            from PIL import Image
                            
                            # Decode TIFF content
                            content_type, content_string = contents.split(',')
                            decoded = base64.b64decode(content_string)
                            
                            # Open TIFF with PIL and convert to PNG
                            img = Image.open(io.BytesIO(decoded))
                            
                            # Convert to PNG in memory
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            png_b64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Create image data structure
                            image_data = {
                                'original_image_b64': png_b64,
                                'numpy_data': np.array(img) if img else None
                            }
                            
                            metadata = {
                                'filename': filename,
                                'format': 'TIFF (converted to PNG)',
                                'size': img.size,
                                'mode': img.mode,
                                'data_type': 'PIL Image'
                            }
                            
                            image_files.append({
                                'image_data': image_data,
                                'metadata': metadata,
                                'filename': filename
                            })
                            uploaded_files.append({
                                'type': 'image',
                                'filename': filename,
                                'status': 'success',
                                'format': 'TIFF->PNG'
                            })
                            print(f"✅ TIFF conversion successful for {filename}")
                            
                        else:
                            # Use existing image processor for other formats
                            if image_processor is None:
                                print("❌ Image processor not available")
                                uploaded_files.append({'type': 'image', 'filename': filename, 'status': 'error', 'error': 'Image processor not loaded'})
                                continue
                                
                            image_data, metadata = image_processor.process_file(contents, filename)
                            image_files.append({
                                'image_data': image_data,
                                'metadata': metadata,
                                'filename': filename
                            })
                            uploaded_files.append({
                                'type': 'image',
                                'filename': filename,
                                'status': 'success',
                                'format': metadata.get('format', 'Unknown')
                            })
                            print(f"✅ Image processing successful for {filename}")
                            
                    except Exception as e:
                        print(f"❌ Image processing failed for {filename}: {e}")
                        uploaded_files.append({
                            'type': 'image',
                            'filename': filename,
                            'status': 'error',
                            'error': str(e)
                        })
                else:
                    print(f"❌ Unsupported file type: {file_ext}")
                    uploaded_files.append({'type': 'unknown', 'filename': filename, 'status': 'error',
                                         'error': f'Unsupported file type: {file_ext}'})
            
            # Check results
            success_files = [f for f in uploaded_files if f['status'] == 'success']
            print(f"🔍 Upload results: {len(success_files)} successful, {len(uploaded_files) - len(success_files)} failed")
            
            if not success_files:
                error_status = dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "⚠️ No valid files were processed. Please check file formats and try again."
                ], color="warning")
                return existing_data, {}, error_status, no_update, no_update, no_update, None
            
            # Create session data
            session_data = {
                'file_type': 'mixed',
                'upload_info': {
                    'num_csv_files': len(csv_files),
                    'num_fits_files': len(fits_files),
                    'num_image_files': len(image_files),
                    'upload_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            # CREATE COMPLETE DISPLAY
            print(f"🔍 Creating display components...")
            main_display_content = create_complete_display(csv_files, fits_files, image_files)
            print(f"🔍 Display components created successfully!")
            
            # Store CSV data for visualization callbacks
            processed_data_store = None
            if csv_files:
                if len(csv_files) == 1:
                    df = csv_files[0]['df']
                    processed_data_store = {
                        'data': df.to_dict('records'),
                        'columns': df.columns.tolist(),
                        'file_type': 'csv',
                        'source_files': [csv_files[0]['filename']]
                    }
                # Handle multiple CSV files if needed
                elif len(csv_files) > 1:
                    # Combine multiple CSV files
                    combined_data, _ = combine_csv_files(csv_files)
                    processed_data_store = {
                        'data': combined_data['data'],
                        'columns': combined_data['columns'],
                        'file_type': 'csv_combined',
                        'source_files': [f['filename'] for f in csv_files]
                    }
            
            # Create status
            status = create_upload_status(uploaded_files, len(csv_files), len(fits_files), len(image_files))
            
            print(f"✅ Upload processing complete!")
            
            return (
                session_data,
                {'processed': True},
                status,
                main_display_content,
                processed_data_store,
                fits_files,
                None   # Clear upload
            )
            
        except Exception as e:
            print(f"❌ Critical error in upload handler: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            logger.error(f"Error in file upload handler: {e}")
            logger.error(traceback.format_exc())
            
            error_status = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"❌ Error processing files: {str(e)}"
            ], color="danger")
            
            return no_update, no_update, error_status, no_update, no_update, no_update, None
    
    @app.callback(
        [Output("welcome-screen", "style"),
         Output("main-display-area", "style", allow_duplicate=True)],
        [Input("session-data", "data")],
        prevent_initial_call=True
    )
    def toggle_display_areas(session_data):
        """Toggle display areas"""
        print(f"🔍 Toggle callback triggered, session_data: {session_data is not None}")
        if session_data and session_data.get('upload_info'):
            print(f"🔍 Hiding welcome screen, showing main area")
            return {"display": "none"}, {"display": "block"}
        else:
            print(f"🔍 Showing welcome screen, hiding main area")
            return {"display": "block"}, {"display": "none"}
    
    @app.callback(
        [Output('session-data', 'data', allow_duplicate=True),
         Output('file-metadata', 'data', allow_duplicate=True),
         Output('processed-data', 'data', allow_duplicate=True),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('main-display-area', 'children', allow_duplicate=True),
         Output('fits-data-store', 'data', allow_duplicate=True)],
        [Input('clear-uploads-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def clear_uploads(n_clicks):
        """Clear uploads"""
        if n_clicks:
            print("🔍 Clearing uploads...")
            return None, None, None, html.Div(), html.Div(), None
        raise PreventUpdate
    
    print("✅ File upload callback registered successfully!")

def create_complete_display(csv_files, fits_files, image_files):
    """Create complete display with full interactive components"""
    content = []
    
    # Add overall header
    content.append(
        dbc.Card([
            dbc.CardBody([
                html.H3([
                    html.I(className="fas fa-rocket me-2"),
                    "AstroAid Analysis Dashboard"
                ]),
                html.P("Your files have been processed and are ready for analysis.", className="text-muted")
            ])
        ], className="mb-4")
    )
    
    # Process CSV files
    if csv_files:
        content.append(create_csv_section(csv_files))
    
    # Process FITS files
    if fits_files:
        content.append(create_fits_section(fits_files))
    
    # Process image files
    if image_files:
        content.append(create_image_section(image_files))
    
    return html.Div(content) if content else html.Div("No files processed")

def create_csv_section(csv_files):
    """Create CSV section with tabbed interface"""
    content = []
    
    content.append(
        html.H4([
            html.I(className="fas fa-table me-2"),
            f"CSV Data ({len(csv_files)} files)"
        ], className="mt-4 mb-3")
    )
    
    # Import and create enhanced data viewer
    try:
        from src.components.data_viewer import create_enhanced_data_viewer
        content.append(create_enhanced_data_viewer())
    except ImportError:
        # Fallback display
        for csv_file in csv_files:
            df = csv_file['df']
            content.append(
                dbc.Alert([
                    html.H5(f"📊 {csv_file['filename']}"),
                    html.P(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"),
                    html.P(f"Columns: {', '.join(df.columns[:5].tolist())}" + ("..." if len(df.columns) > 5 else ""))
                ], color="success", className="mb-3")
            )
    
    return html.Div(content)

def create_fits_section(fits_files):
    """Create FITS section with full interactive controls"""
    print(f"🔍 Creating FITS section for {len(fits_files)} files")
    
    content = []
    
    # Section header
    content.append(
        html.H4([
            html.I(className="fas fa-satellite me-2"),
            f"FITS Astronomical Data ({len(fits_files)} files)"
        ], className="mt-4 mb-3")
    )
    
    # Process each FITS file
    for file_idx, fits_file in enumerate(fits_files):
        filename = fits_file.get('filename', f'File_{file_idx}')
        hdu_list = fits_file.get('hdu_list', [])
        
        print(f"🔍 Processing FITS file: {filename} with {len(hdu_list)} HDUs")
        
        # File header
        content.append(
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        html.I(className="fas fa-file me-2"),
                        filename
                    ], className="mb-0"),
                    html.Small(f"{len(hdu_list)} HDUs detected", className="text-muted")
                ])
            ], className="mb-3")
        )
        
        # Process each HDU
        for hdu_idx, hdu in enumerate(hdu_list):
            hdu_card = create_interactive_hdu_card(hdu, filename, file_idx, hdu_idx)
            content.append(hdu_card)
    
    return html.Div(content)

def create_interactive_hdu_card(hdu, filename, file_idx, hdu_idx):
    """Create interactive HDU card with complete controls"""
    hdu_type = hdu.get('type', 'Unknown')
    
    # Create unique ID for callbacks
    safe_filename = (filename.replace(' ', '-').replace('.', '-').replace('(', '')
                    .replace(')', '').replace('/', '-'))[:30]
    unique_id = f"{safe_filename}-{file_idx}-{hdu_idx}"
    
    card_content = []
    
    # HDU Header
    card_content.append(
        dbc.CardHeader([
            html.H6([
                html.I(className=f"fas fa-{'image' if hdu_type == 'IMAGE' else 'table'} me-2"),
                f"HDU {hdu_idx}: {hdu_type}"
            ], className="mb-0"),
            html.Small(hdu.get('info', ''), className="text-muted")
        ])
    )
    
    # HDU Body
    body_content = []
    
    if hdu_type == 'IMAGE' and hdu.get('image_b64'):
        # IMAGE HDU with FULL interactive controls
        raw_data = hdu.get('raw_image_data')
        
        body_content.append(
            dbc.Row([
                # Image Display
                dbc.Col([
                    # Original static image
                    html.Div([
                        html.H6("📸 Original FITS Image", className="mb-2"),
                        html.Img(
                            src=f"data:image/png;base64,{hdu['image_b64']}",
                            style={'width': '100%', 'maxHeight': '300px', 'objectFit': 'contain',
                                   'border': '1px solid #dee2e6', 'borderRadius': '4px'},
                            className="mb-3"
                        )
                    ]),
                    
                    # Interactive processed image
                    html.Div([
                        html.H6("🔬 Interactive Processing", className="mb-2"),
                        dcc.Graph(
                            id={'type': 'fits-interactive-plot', 'index': unique_id},
                            figure={
                                'data': [{
                                    'type': 'heatmap',
                                    'z': raw_data if raw_data else [[0, 1], [2, 3]],
                                    'colorscale': 'Viridis',
                                    'showscale': True,
                                    'hovertemplate': 'X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f}<extra></extra>'
                                }],
                                'layout': {
                                    'title': f'{filename} HDU {hdu_idx} - Interactive View',
                                    'xaxis': {'title': 'X (pixels)', 'showgrid': False},
                                    'yaxis': {'title': 'Y (pixels)', 'showgrid': False, 'autorange': 'reversed'},
                                    'dragmode': 'zoom',
                                    'height': 400
                                }
                            },
                            config={'displayModeBar': True, 'displaylogo': False}
                        ),
                        
                        # Status
                        html.Div(
                            id={'type': 'fits-region-info', 'index': unique_id},
                            className="mt-2 p-2 bg-light rounded",
                            children=[html.Small("✅ FITS image loaded - Use controls to apply filters", className="text-success")]
                        )
                    ])
                ], width=8),
                
                # COMPLETE CONTROLS COLUMN
                dbc.Col([
                    # Filter Controls
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
                                id={'type': 'fits-filter', 'index': unique_id},
                                options=[
                                    {'label': '🚫 None', 'value': 'none'},
                                    {'label': '✨ Brightness', 'value': 'brightness'},
                                    {'label': '🎚️ Contrast', 'value': 'contrast'},
                                    {'label': '🌀 Gaussian Smooth', 'value': 'gaussian'},
                                    {'label': '🔹 Median Filter', 'value': 'median'},
                                    {'label': '🔍 Sobel Edge', 'value': 'sobel'}
                                ],
                                value='none',
                                className="mb-3"
                            ),
                            
                            html.Label("Filter Strength:", className="form-label fw-bold"),
                            dcc.Slider(
                                id={'type': 'fits-strength', 'index': unique_id},
                                min=0.1, max=3.0, value=1.0, step=0.1,
                                marks={0.5: '0.5', 1: '1', 2: '2', 3: '3'},
                                className="mb-3"
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Colormap Controls
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
                                id={'type': 'fits-colormap', 'index': unique_id},
                                options=[
                                    {'label': '🌌 Viridis', 'value': 'Viridis'},
                                    {'label': '🌟 Plasma', 'value': 'Plasma'},
                                    {'label': '🌋 Inferno', 'value': 'Inferno'},
                                    {'label': '🔮 Magma', 'value': 'Magma'},
                                    {'label': '⚫ Greys', 'value': 'Greys'},
                                    {'label': '🔥 Hot', 'value': 'Hot'},
                                    {'label': '❄️ Blues', 'value': 'Blues'}
                                ],
                                value='Viridis',
                                className="mb-3"
                            ),
                            
                            html.Label("Intensity Scale:", className="form-label fw-bold"),
                            dcc.Dropdown(
                                id={'type': 'fits-scale', 'index': unique_id},
                                options=[
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Logarithmic', 'value': 'log'},
                                    {'label': 'Square Root', 'value': 'sqrt'},
                                    {'label': 'Asinh', 'value': 'asinh'}
                                ],
                                value='linear',
                                className="mb-3"
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Action Buttons
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button([
                                html.I(className="fas fa-magic me-2"),
                                "Apply Processing"
                            ], id={'type': 'fits-apply', 'index': unique_id},
                               color="primary", size="sm", className="w-100 mb-2"),
                            
                            dbc.Button([
                                html.I(className="fas fa-undo me-2"),
                                "Reset Original"
                            ], id={'type': 'fits-reset', 'index': unique_id},
                               color="secondary", size="sm", className="w-100")
                        ])
                    ])
                ], width=4)
            ])
        )
        
        # Store original data for callbacks
        body_content.append(
            dcc.Store(
                id={'type': 'fits-original-data', 'index': unique_id},
                data={
                    'image_data': raw_data,
                    'header': hdu.get('header', {}),
                    'filename': filename,
                    'hdu_index': hdu_idx
                }
            )
        )
    
    elif hdu_type == 'TABLE':
        # Table HDU
        table_data = hdu.get('table_data')
        if table_data:
            body_content.append(
                html.Div([
                    html.H6("📊 Table Data", className="mb-2"),
                    html.P(f"Rows: {len(table_data.get('data', []))}, Columns: {len(table_data.get('columns', []))}",
                           className="text-muted small")
                ])
            )
    else:
        body_content.append(
            html.P("Header-only HDU", className="text-muted text-center py-3")
        )
    
    card_content.append(dbc.CardBody(body_content))
    
    return dbc.Card(card_content, className="mb-4")

def create_image_section(image_files):
    """Create image section with CORRECTED IDs to match existing callbacks"""
    print(f"🔍 Creating image section for {len(image_files)} files")
    
    content = []
    
    # Section header
    content.append(
        html.H4([
            html.I(className="fas fa-image me-2"),
            f"Image Analysis ({len(image_files)} files)"
        ], className="mt-4 mb-3")
    )
    
    # Process each image file
    for img_idx, image_file in enumerate(image_files):
        filename = image_file.get('filename', f'Image_{img_idx}')
        image_data = image_file.get('image_data', {})
        metadata = image_file.get('metadata', {})
        
        print(f"🔍 Creating image display for: {filename}")
        
        # Get the base64 image data
        original_b64 = image_data.get('original_image_b64', '')
        
        image_card = dbc.Card([
            # Card Header
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-microscope me-2"),  # Updated icon
                    filename
                ], className="mb-0"),
                html.Small([
                    f"Format: {metadata.get('format', 'Unknown')} • ",
                    f"Size: {metadata.get('size', 'Unknown')}"
                ], className="text-muted")
            ]),
            
            # Card Body with CORRECTED IDs
            dbc.CardBody([
                dbc.Row([
                    # Image Display
                    dbc.Col([
                        html.H6("🖼️ Image", className="mb-2"),
                        html.Img(
                            src=f"data:image/png;base64,{original_b64}" if original_b64 else "",
                            style={
                                'width': '100%',
                                'maxHeight': '600px',
                                'objectFit': 'contain',
                                'border': '2px solid #dee2e6',
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                            },
                            className="mb-3",
                            id={'type': 'processed-image', 'index': img_idx}  # ✅ CORRECTED ID
                        ) if original_b64 else html.P("No image data available", className="text-muted"),
                        
                        # Processing status indicator
                        html.Div(
                            id={'type': 'processing-status', 'index': img_idx},
                            className="mt-2 text-center",
                            children=[
                                dbc.Badge("Original", color="secondary", className="me-2"),
                                html.Small("Ready for processing", className="text-muted")
                            ]
                        )
                    ], width=8),
                    
                    # CORRECTED CONTROLS WITH MATCHING IDs
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
                                    id={'type': 'filter-type', 'index': img_idx},  # ✅ CORRECTED ID
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
                                    id={'type': 'filter-strength', 'index': img_idx},  # ✅ CORRECTED ID
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
                                    id={'type': 'colormap', 'index': img_idx},  # ✅ CORRECTED ID
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
                                id={'type': 'apply-processing', 'index': img_idx},  # ✅ CORRECTED ID
                                color="primary",
                                size="sm",
                                className="w-100 mb-2"),
                                
                                dbc.Button([
                                    html.I(className="fas fa-undo me-2"),
                                    "Reset to Original"
                                ],
                                id={'type': 'reset-processing', 'index': img_idx},  # ✅ CORRECTED ID
                                color="secondary",
                                size="sm",
                                className="w-100 mb-2"),
                                
                                dbc.Button([
                                    html.I(className="fas fa-download me-2"),
                                    "Export Processed"
                                ],
                                id={'type': 'export-image', 'index': img_idx},  # ✅ CORRECTED ID
                                color="success",
                                size="sm",
                                className="w-100")
                            ])
                        ])
                    ], width=4)
                ])
            ]),
            
            # Hidden data store for original image
            dcc.Store(
                id={'type': 'original-image-data', 'index': img_idx},  # ✅ CORRECTED ID
                data={
                    'numpy_data': image_data.get('numpy_data', []),
                    'filename': filename,
                    'original_b64': image_data.get('original_image_b64', '')
                }
            )
            
        ], className="mb-4")
        
        content.append(image_card)
    
    return html.Div(content)

def combine_csv_files(csv_files):
    """Combine multiple CSV files"""
    try:
        all_columns = set()
        dataframes = []
        
        # Collect columns and dataframes
        for csv_file in csv_files:
            df = csv_file['df']
            all_columns.update(df.columns)
            dataframes.append(df)
        
        # Align columns and combine
        aligned_dfs = []
        for i, df in enumerate(dataframes):
            aligned_df = df.reindex(columns=list(all_columns), fill_value=None)
            aligned_df['_source_file'] = csv_files[i]['filename']
            aligned_dfs.append(aligned_df)
        
        combined_df = pd.concat(aligned_dfs, ignore_index=True, sort=False)
        
        combined_data = {
            'data': combined_df.to_dict('records'),
            'columns': combined_df.columns.tolist()
        }
        
        combined_metadata = {
            'filename': f"Combined_{len(csv_files)}_files",
            'total_rows': len(combined_df),
            'source_files': [f['filename'] for f in csv_files]
        }
        
        return combined_data, combined_metadata
        
    except Exception as e:
        logger.error(f"Error combining CSV files: {e}")
        raise

def create_upload_status(uploaded_files, csv_count, fits_count, image_count):
    """Create upload status"""
    success_files = [f for f in uploaded_files if f['status'] == 'success']
    error_files = [f for f in uploaded_files if f['status'] == 'error']
    
    status_items = []
    
    for success_file in success_files:
        status_items.append(
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"✅ Successfully loaded: {success_file['filename']}"
            ], color="success")
        )
    
    for error_file in error_files:
        status_items.append(
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"❌ Failed: {error_file['filename']} - {error_file['error']}"
            ], color="danger")
        )
    
    return html.Div(status_items)
