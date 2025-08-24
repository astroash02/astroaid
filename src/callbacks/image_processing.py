"""
Interactive image processing callbacks with BULLETPROOF error handling
"""

from dash import Input, Output, State, callback, MATCH, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import base64
import logging
from src.processors.image_processor import ImageProcessor

logger = logging.getLogger(__name__)
image_processor = ImageProcessor()

def register_image_processing_callbacks(app):
    """Register callbacks with comprehensive error handling"""
    
    @app.callback(
        Output({'type': 'processed-image', 'index': MATCH}, 'src'),
        [Input({'type': 'apply-processing', 'index': MATCH}, 'n_clicks')],
        [State({'type': 'filter-type', 'index': MATCH}, 'value'),
         State({'type': 'filter-strength', 'index': MATCH}, 'value'),
         State({'type': 'colormap', 'index': MATCH}, 'value'),
         State({'type': 'original-image-data', 'index': MATCH}, 'data')],
        prevent_initial_call=True
    )
    def apply_interactive_processing(n_clicks, filter_type, strength, colormap, original_data):
        """Apply processing with bulletproof error handling"""
        if not n_clicks or not original_data:
            raise PreventUpdate
            
        try:
            # Safely reconstruct image array
            numpy_data = original_data.get('numpy_data')
            if not numpy_data:
                raise PreventUpdate
                
            image_array = np.array(numpy_data, dtype=np.uint8)
            
            # Apply filter if selected
            if filter_type and filter_type != 'none':
                image_array = image_processor.apply_advanced_filter(image_array, filter_type, strength=strength)
            
            # Apply colormap if selected
            if colormap and colormap != 'original':
                image_array = image_processor.apply_enhanced_colormap(image_array, colormap)
            
            # Convert to base64
            processed_b64 = image_processor._create_display_image(image_array, "processed")
            
            if processed_b64:
                return f"data:image/png;base64,{processed_b64}"
            else:
                raise PreventUpdate
                
        except Exception as e:
            logger.error(f"Error in interactive processing: {e}")
            # Return original image on error
            original_b64 = original_data.get('original_b64', '')
            if original_b64:
                return f"data:image/png;base64,{original_b64}"
            raise PreventUpdate
    
    @app.callback(
        Output({'type': 'processing-status', 'index': MATCH}, 'children'),
        [Input({'type': 'filter-type', 'index': MATCH}, 'value'),
         Input({'type': 'colormap', 'index': MATCH}, 'value')],
        prevent_initial_call=True
    )
    def update_processing_status(filter_type, colormap):
        """Update status badges"""
        badges = []
        
        if filter_type and filter_type != 'none':
            badges.append(dbc.Badge(f"Filter: {filter_type}", color="info", className="me-2"))
        
        if colormap and colormap != 'original':
            badges.append(dbc.Badge(f"Colormap: {colormap}", color="success", className="me-2"))
            
        if not badges:
            badges.append(dbc.Badge("Original", color="secondary", className="me-2"))
            
        return badges + [html.Small("Ready", className="text-muted")]
    
    @app.callback(
        Output({'type': 'processed-image', 'index': MATCH}, 'src', allow_duplicate=True),
        [Input({'type': 'reset-processing', 'index': MATCH}, 'n_clicks')],
        [State({'type': 'original-image-data', 'index': MATCH}, 'data')],
        prevent_initial_call=True
    )
    def reset_to_original(n_clicks, original_data):
        """Reset to original image"""
        if n_clicks and original_data:
            original_b64 = original_data.get('original_b64', '')
            if original_b64:
                return f"data:image/png;base64,{original_b64}"
        raise PreventUpdate
