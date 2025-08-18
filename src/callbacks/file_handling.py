"""
Callbacks for handling file uploads and processing
Enhanced with FITS image display functionality
"""

from dash import Input, Output, State, callback, html, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
import traceback
import pandas as pd
import base64
import io
from typing import List, Dict, Any
from src.processors.csv_processor import CSVProcessor
from src.processors.fits_processor import FITSProcessor
from dash import dash_table

logger = logging.getLogger(__name__)

# Initialize processors
csv_processor = CSVProcessor()
fits_processor = FITSProcessor()

def decode_file_content(content_string):
    """Safely decode uploaded file content"""
    try:
        content_type, content_string = content_string.split(',')
        decoded = base64.b64decode(content_string)
        return decoded
    except Exception as e:
        logger.error(f"Error decoding file content: {e}")
        raise

def process_csv_file_direct(contents, filename):
    """Direct CSV processing with robust error handling"""
    try:
        decoded = decode_file_content(contents)
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                csv_string = decoded.decode(encoding)
                df = pd.read_csv(io.StringIO(csv_string))
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        else:
            # If all encodings fail, try with error handling
            csv_string = decoded.decode('utf-8', errors='replace')
            df = pd.read_csv(io.StringIO(csv_string))

        # Clean the dataframe
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Replace infinite values
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        
        # Create metadata
        metadata = {
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'file_type': 'csv',
            'encoding': encoding,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'text_columns': len(df.select_dtypes(include=['object']).columns),
            'column_names': df.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict()
        }
        
        logger.info(f"Successfully processed CSV {filename}: {df.shape}")
        return df, metadata
        
    except Exception as e:
        logger.error(f"Error processing CSV {filename}: {e}")
        raise

def create_fits_display(fits_files: List[Dict]) -> html.Div:
    """Create comprehensive FITS display with images"""
    if not fits_files:
        return html.Div([
            dbc.Alert("No FITS files to display", color="warning")
        ])

    content = []
    
    # Header
    total_hdus = sum(len(f.get('hdu_list', [])) for f in fits_files)
    content.append(
        dbc.Card([
            dbc.CardBody([
                html.H3([
                    html.I(className="fas fa-satellite me-2"),
                    f"FITS Analysis - {len(fits_files)} file(s), {total_hdus} HDUs"
                ]),
                html.Div([
                    dbc.Badge(f"{len(fits_files)} FITS files", color="info", className="me-2"),
                    dbc.Badge(f"{total_hdus} HDUs total", color="success", className="me-2"),
                ])
            ])
        ], className="mb-4")
    )
    
    # Process each FITS file
    for file_idx, fits_file in enumerate(fits_files):
        filename = fits_file.get('filename', f'File_{file_idx}')
        hdu_list = fits_file.get('hdu_list', [])
        
        if not hdu_list:
            continue
        
        # File header
        content.append(
            html.H4([
                html.I(className="fas fa-file me-2"),
                filename
            ], className="mt-4 mb-3")
        )
        
        # Process each HDU
        hdu_rows = []
        for hdu_idx in range(0, len(hdu_list), 2):  # Two HDUs per row
            hdu_cols = []
            for i in range(2):
                if hdu_idx + i < len(hdu_list):
                    hdu = hdu_list[hdu_idx + i]
                    hdu_card = create_hdu_card(hdu, filename)
                    hdu_cols.append(dbc.Col([hdu_card], width=6, className="mb-3"))
            
            if hdu_cols:
                hdu_rows.append(dbc.Row(hdu_cols))
        
        content.extend(hdu_rows)

    return html.Div(content)

def create_hdu_card(hdu: Dict, filename: str) -> dbc.Card:
    """Create individual HDU card with images and data"""
    hdu_type = hdu.get('type', 'Unknown')
    hdu_index = hdu.get('index', 0)
    hdu_info = hdu.get('info', '')
    
    # Card header
    card_header = dbc.CardHeader([
        html.H6([
            html.I(className=f"fas fa-{'image' if hdu_type == 'IMAGE' else 'table' if hdu_type == 'TABLE' else 'file-alt'} me-2"),
            f"HDU {hdu_index}: {hdu_type}"
        ], className="mb-0"),
        html.Small(hdu_info, className="text-muted")
    ])
    
    # Card body content
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
    
    # Display table data with proper DataFrame check
    table_data = hdu.get('table_data')
    if table_data is not None:
        try:
            if isinstance(table_data, dict) and 'data' in table_data:
                preview_data = table_data['data'][:5] if len(table_data['data']) > 5 else table_data['data']
                columns = table_data.get('columns', [])
                preview_columns = [{"name": col, "id": col} for col in columns[:6]]
                
                mini_table = dash_table.DataTable(
                    data=preview_data,
                    columns=preview_columns,
                    style_table={'overflowX': 'auto', 'maxHeight': '200px'},
                    style_cell={'textAlign': 'left', 'fontSize': '11px', 'padding': '4px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    page_size=5
                )
                
                card_body_content.extend([
                    html.H6("📊 Table Data Preview:", className="mt-3 mb-2"),
                    html.P(f"{len(table_data['data'])} rows × {len(columns)} columns",
                           className="small text-muted mb-2"),
                    mini_table
                ])
        except Exception as table_error:
            logger.debug(f"Error processing table data: {table_error}")
    
    # Display header information
    if hdu.get('header') and isinstance(hdu['header'], dict):
        header_info = hdu['header']
        important_keys = ['OBJECT', 'DATE-OBS', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME', 'NAXIS1', 'NAXIS2']
        important_header = {k: v for k, v in header_info.items() if k in important_keys and v}
        
        if important_header:
            header_items = []
            for key, value in list(important_header.items())[:6]:
                header_items.append(
                    html.Tr([
                        html.Td(html.Code(key), style={'width': '30%'}),
                        html.Td(str(value)[:50] + ('...' if len(str(value)) > 50 else ''))
                    ])
                )
            
            header_table = dbc.Table([
                html.Tbody(header_items)
            ], size="sm", className="mb-0")
            
            card_body_content.extend([
                html.H6("📋 Header Keywords:", className="mt-3 mb-2"),
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

def create_csv_display(df: pd.DataFrame, metadata: Dict, source_files: List[str]) -> html.Div:
    """Create CSV data display"""
    try:
        from src.components.data_viewer import create_enhanced_data_viewer
        
        is_combined = len(source_files) > 1
        header_title = f"Combined CSV Data ({len(source_files)} files)" if is_combined else f"CSV Data: {source_files[0] if source_files else 'Unknown'}"
        
        return html.Div([
            # File info header
            dbc.Card([
                dbc.CardBody([
                    html.H4([
                        html.I(className="fas fa-database me-2"),
                        header_title
                    ]),
                    # File info badges
                    dbc.Row([
                        dbc.Col([
                            dbc.Badge(f"{len(df):,} rows", color="info", className="me-2"),
                            dbc.Badge(f"{len(df.columns):,} columns", color="success", className="me-2"),
                            dbc.Badge(f"{len(source_files)} file(s)" if is_combined else "Single file",
                                     color="secondary", className="me-2"),
                        ], width=12)
                    ], className="mb-3"),
                ])
            ], className="mb-4"),
            
            # Enhanced data viewer
            create_enhanced_data_viewer()
        ])
        
    except ImportError:
        return dbc.Alert([
            html.H4("CSV Data Loaded Successfully"),
            html.P(f"Files: {', '.join(source_files)}"),
            html.P(f"Data: {len(df):,} rows × {len(df.columns):,} columns"),
        ], color="info")

def register_callbacks(app):
    """Register all file handling callbacks"""
    
    @app.callback(
        [
            Output('session-data', 'data', allow_duplicate=True),
            Output('file-metadata', 'data', allow_duplicate=True),
            Output('upload-status', 'children', allow_duplicate=True),
            Output('main-display-area', 'children', allow_duplicate=True),
            Output('processed-data', 'data', allow_duplicate=True),
            Output('fits-data-store', 'data', allow_duplicate=True)
        ],
        [Input('file-upload', 'contents')],
        [
            State('file-upload', 'filename'),
            State('file-upload', 'last_modified'),
            State('session-data', 'data')
        ],
        prevent_initial_call=True
    )
    def handle_file_upload(list_of_contents, list_of_names, list_of_dates, existing_data):
        """Handle file upload and processing with FITS image display"""
        if not list_of_contents:
            if existing_data:
                raise PreventUpdate
            else:
                return None, None, None, html.Div(), None, None

        try:
            uploaded_files = []
            csv_files = []
            fits_files = []
            
            # Process all uploaded files
            for i, contents in enumerate(list_of_contents):
                filename = list_of_names[i] if list_of_names and i < len(list_of_names) else f"unknown_{i}"
                file_ext = filename.lower().split('.')[-1]
                
                logger.info(f"Processing uploaded file {i+1}/{len(list_of_contents)}: {filename}")
                
                if file_ext in ['csv', 'tsv', 'dat', 'txt']:
                    try:
                        try:
                            df, metadata = csv_processor.process_file(contents, filename)
                        except Exception as processor_error:
                            logger.warning(f"CSV processor failed for {filename}, trying direct processing: {processor_error}")
                            df, metadata = process_csv_file_direct(contents, filename)
                        
                        csv_files.append({'df': df, 'metadata': metadata, 'filename': filename})
                        uploaded_files.append({'type': 'csv', 'filename': filename, 'status': 'success', 'rows': len(df)})
                        
                    except Exception as e:
                        logger.error(f"Error processing CSV {filename}: {e}")
                        uploaded_files.append({'type': 'csv', 'filename': filename, 'status': 'error', 'error': str(e)})
                
                elif file_ext in ['fits', 'fit', 'fts']:
                    try:
                        hdu_list, file_metadata = fits_processor.process_file(contents, filename)
                        fits_files.append({'hdu_list': hdu_list, 'metadata': file_metadata, 'filename': filename})
                        uploaded_files.append({'type': 'fits', 'filename': filename, 'status': 'success', 'hdus': len(hdu_list)})
                        
                    except Exception as e:
                        logger.error(f"Error processing FITS {filename}: {e}")
                        uploaded_files.append({'type': 'fits', 'filename': filename, 'status': 'error', 'error': str(e)})
                
                else:
                    uploaded_files.append({'type': 'unknown', 'filename': filename, 'status': 'error', 
                                         'error': f'Unsupported file type: {file_ext}'})

            # Create session data
            session_data = create_lightweight_session_data(csv_files, fits_files)
            
            # Handle display and data storage
            processed_data_store = None
            combined_metadata = None
            main_display_content = html.Div()
            
            if len(csv_files) > 1:
                # Combine multiple CSV files
                combined_data, combined_metadata = combine_csv_files(csv_files)
                processed_data_store = {
                    'data': combined_data['data'],
                    'columns': combined_data['columns'],
                    'file_type': 'csv_combined',
                    'source_files': [f['filename'] for f in csv_files]
                }
                df = pd.DataFrame(combined_data['data'])
                main_display_content = create_csv_display(df, combined_metadata, [f['filename'] for f in csv_files])
                
            elif len(csv_files) == 1:
                # Single CSV file - FIXED: Access list element first, then the key
                df = csv_files[0]['df']
                if csv_files and isinstance(csv_files, list) and len(csv_files) > 0:
                    combined_metadata = csv_files[0]['metadata']
                else:
                    combined_metadata = {}

                # ONLY LARGE DATA GOES TO processed-data store
                processed_data_store = {
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'file_type': 'csv',
                    'source_files': [csv_files[0]['filename']]
                }
                main_display_content = create_csv_display(df, combined_metadata, [csv_files[0]['filename']])
            elif len(fits_files) > 0:
                # FITS files - CREATE THE DISPLAY HERE
                combined_metadata = {
                    'total_fits_files': len(fits_files),
                    'total_hdus': sum(len(f['hdu_list']) for f in fits_files),
                    'fits_file_info': [f.get('metadata', {}) for f in fits_files]
                }
                
                # Create the FITS display
                main_display_content = create_fits_display(fits_files)
                
            else:
                raise ValueError("No valid files were processed successfully")

            # Create status message
            status = create_upload_status(uploaded_files, len(csv_files), len(fits_files))
            
            # Convert any DataFrames in fits_files to JSON-serializable format
            serializable_fits_files = []
            for fits_file in fits_files:
                serializable_fits_file = {
                    'filename': fits_file['filename'],
                    'metadata': fits_file['metadata'],
                    'hdu_list': []
                }
                
                for hdu in fits_file['hdu_list']:
                    serializable_hdu = dict(hdu)
                    
                    # Convert table_data if it's a DataFrame
                    if 'table_data' in serializable_hdu and hasattr(serializable_hdu['table_data'], 'to_dict'):
                        df = serializable_hdu['table_data']
                        serializable_hdu['table_data'] = {
                            'data': df.to_dict('records'),
                            'columns': df.columns.tolist()
                        }
                    
                    serializable_fits_file['hdu_list'].append(serializable_hdu)
                
                serializable_fits_files.append(serializable_fits_file)
            
            return (
                session_data,           # session-data
                combined_metadata,      # file-metadata
                status,                 # upload-status
                main_display_content,   # main-display-area
                processed_data_store,   # processed-data
                serializable_fits_files # fits-data-store
            )
            
        except Exception as e:
            logger.error(f"Error in file upload handler: {e}")
            logger.error(traceback.format_exc())
            
            error_status = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"❌ Error processing files: {str(e)}"
            ], color="danger")
            
            return no_update, no_update, error_status, no_update, no_update, no_update

    @app.callback(
        [Output("welcome-screen", "style"),
         Output("main-display-area", "style", allow_duplicate=True)],
        [Input("session-data", "data")],
        prevent_initial_call=True
    )
    def toggle_display_areas(session_data):
        """Show/hide areas based on data availability"""
        if session_data and session_data.get('upload_info'):
            return {"display": "none"}, {"display": "block"}
        else:
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
        """Clear all uploaded data and reset UI"""
        if n_clicks:
            return None, None, None, html.Div(), html.Div(), None
        raise PreventUpdate

def create_lightweight_session_data(csv_files: List[Dict], fits_files: List[Dict]) -> Dict:
    """Create minimal session data to avoid QuotaExceededError"""
    if csv_files:
        return {
            'file_type': 'csv_combined' if len(csv_files) > 1 else 'csv',
            'upload_info': {
                'num_csv_files': len(csv_files),
                'num_fits_files': len(fits_files),
                'csv_filenames': [f['filename'] for f in csv_files],
                'fits_filenames': [f['filename'] for f in fits_files],
                'total_csv_rows': sum(len(f['df']) for f in csv_files),
                'total_csv_columns': sum(len(f['df'].columns) for f in csv_files),
                'upload_timestamp': pd.Timestamp.now().isoformat()
            }
        }
    elif fits_files:
        return {
            'file_type': 'fits_multiple' if len(fits_files) > 1 else 'fits',
            'upload_info': {
                'num_csv_files': 0,
                'num_fits_files': len(fits_files),
                'fits_filenames': [f['filename'] for f in fits_files],
                'total_fits_hdus': sum(len(f['hdu_list']) for f in fits_files),
                'upload_timestamp': pd.Timestamp.now().isoformat()
            }
        }
    else:
        return {}

def combine_csv_files(csv_files: List[Dict]) -> tuple:
    """Combine multiple CSV files into one dataset"""
    try:
        dataframes = []
        all_columns = set()
        
        # Collect all unique columns
        for csv_file in csv_files:
            df = csv_file['df']
            all_columns.update(df.columns)
            dataframes.append(df)
        
        # Align all dataframes to have the same columns
        aligned_dfs = []
        for i, df in enumerate(dataframes):
            aligned_df = df.reindex(columns=list(all_columns), fill_value=None)
            # Add source file column
            aligned_df['_source_file'] = csv_files[i]['filename']
            aligned_dfs.append(aligned_df)
        
        # Combine all dataframes
        combined_df = pd.concat(aligned_dfs, ignore_index=True, sort=False)
        
        # Generate combined metadata
        combined_metadata = {
            'filename': f"Combined_{len(csv_files)}_files",
            'file_type': 'csv_combined',
            'source_files': [f['filename'] for f in csv_files],
            'total_rows': len(combined_df),
            'total_columns': len(combined_df.columns) - 1,
            'source_file_rows': {f['filename']: len(f['df']) for f in csv_files},
            'encoding': 'combined',
            'separator': 'combined',
            'numeric_columns': len(combined_df.select_dtypes(include=['number']).columns),
            'text_columns': len(combined_df.select_dtypes(include=['object']).columns) - 1,
            'memory_usage_mb': combined_df.memory_usage().sum() / (1024 * 1024),
            'column_names': [col for col in combined_df.columns if col != '_source_file'],
            'data_types': {k: str(v) for k, v in combined_df.dtypes.to_dict().items() if k != '_source_file'},
            'null_counts': {k: int(v) for k, v in combined_df.isnull().sum().to_dict().items() if k != '_source_file'},
        }
        
        combined_data = {
            'data': combined_df.to_dict('records'),
            'columns': combined_df.columns.tolist(),
            'combined_info': {
                'total_files': len(csv_files),
                'source_files': [f['filename'] for f in csv_files],
                'individual_rows': [len(f['df']) for f in csv_files]
            }
        }
        
        logger.info(f"Successfully combined {len(csv_files)} CSV files: {combined_df.shape}")
        return combined_data, combined_metadata
        
    except Exception as e:
        logger.error(f"Error combining CSV files: {e}")
        raise

def create_upload_status(uploaded_files: List[Dict], csv_count: int, fits_count: int) -> html.Div:
    """Create status message for uploaded files"""
    success_files = [f for f in uploaded_files if f['status'] == 'success']
    error_files = [f for f in uploaded_files if f['status'] == 'error']
    
    status_items = []
    
    # Success message
    if success_files:
        if csv_count > 1:
            total_rows = sum(f.get('rows', 0) for f in success_files if f['type'] == 'csv')
            status_items.append(
                dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    f"✅ Successfully combined {csv_count} CSV files into one dataset ({total_rows:,} total rows)"
                ], color="success")
            )
        else:
            for success_file in success_files:
                if success_file['type'] == 'csv':
                    status_items.append(
                        dbc.Alert([
                            html.I(className="fas fa-check-circle me-2"),
                            f"✅ Successfully loaded CSV file: {success_file['filename']} ({success_file.get('rows', 0):,} rows)"
                        ], color="success")
                    )
                elif success_file['type'] == 'fits':
                    status_items.append(
                        dbc.Alert([
                            html.I(className="fas fa-check-circle me-2"),
                            f"✅ Successfully loaded FITS file: {success_file['filename']} ({success_file.get('hdus', 0)} HDUs)"
                        ], color="success")
                    )
    
    # Error messages
    for error_file in error_files:
        status_items.append(
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"❌ Failed to load {error_file['filename']}: {error_file['error']}"
            ], color="danger")
        )
    
    return html.Div(status_items)
