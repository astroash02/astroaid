"""
Enhanced FITS Processing Callbacks - FIXED VERSION
"""

from dash import Input, Output, State, callback, MATCH, html, ctx
from dash.exceptions import PreventUpdate
import numpy as np
import logging

logger = logging.getLogger(__name__)

def register_fits_processing_callbacks(app):
    """Register FITS processing callbacks - FIXED VERSION"""
    
    @app.callback(
        [Output({'type': 'fits-interactive-plot', 'index': MATCH}, 'figure'),
         Output({'type': 'fits-region-info', 'index': MATCH}, 'children')],
        [Input({'type': 'fits-apply', 'index': MATCH}, 'n_clicks'),
         Input({'type': 'fits-reset', 'index': MATCH}, 'n_clicks')],
        [State({'type': 'fits-filter', 'index': MATCH}, 'value'),
         State({'type': 'fits-strength', 'index': MATCH}, 'value'),
         State({'type': 'fits-colormap', 'index': MATCH}, 'value'),
         State({'type': 'fits-scale', 'index': MATCH}, 'value'),
         State({'type': 'fits-original-data', 'index': MATCH}, 'data')],
        prevent_initial_call=True
    )
    def process_fits_image_enhanced(apply_clicks, reset_clicks, filter_type, strength,
                                  colormap, scale_type, original_data):
        if not original_data:
            raise PreventUpdate
        
        # Determine which button was clicked
        trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else None
        
        try:
            image_data = original_data.get('image_data', None)
            if image_data is None or not isinstance(image_data, list):
                raise PreventUpdate
            
            # Convert to numpy array
            arr = np.array(image_data, dtype=np.float64)
            
            if 'fits-reset' in str(trigger):
                # Reset to original image
                figure = create_fits_figure(arr, 'Original FITS Image', 'Viridis')
                status = html.Small("↩️ Reset to original FITS image", className="text-info")
            elif 'fits-apply' in str(trigger) and apply_clicks:
                # Apply processing
                logger.info(f"Applying {filter_type} filter with strength {strength}")
                
                # Apply filter
                processed_data = apply_astronomical_filter(arr, filter_type or 'none', strength or 1.0)
                
                # Apply scaling
                if scale_type and scale_type != 'linear':
                    processed_data = apply_enhanced_intensity_scaling(processed_data, scale_type)
                
                # Create figure
                title = f'FITS Processing: {filter_type.title()} + {colormap} + {scale_type}'
                figure = create_fits_figure(processed_data, title, colormap or 'Viridis')
                status = html.Small("✅ FITS processing applied successfully!", className="text-success")
            else:
                raise PreventUpdate
            
            return figure, status
            
        except Exception as e:
            logger.error(f"Error in FITS processing: {e}")
            raise PreventUpdate

def create_fits_figure(data_array, title, colormap):
    """Create FITS figure"""
    figure = {
        'data': [{
            'type': 'heatmap',
            'z': data_array.tolist(),
            'colorscale': colormap,
            'showscale': True,
            'hovertemplate': 'X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f}<extra></extra>'
        }],
        'layout': {
            'title': title,
            'xaxis': {
                'title': 'X (pixels)',
                'showgrid': False,
                'zeroline': False
            },
            'yaxis': {
                'title': 'Y (pixels)',
                'showgrid': False,
                'zeroline': False,
                'autorange': 'reversed'
            },
            'dragmode': 'zoom',
            'height': 500,
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60}
        }
    }
    return figure

def apply_astronomical_filter(image: np.ndarray, filter_type: str, strength: float) -> np.ndarray:
    """Apply astronomical filters"""
    try:
        if filter_type == 'none':
            return image
        elif filter_type == 'brightness':
            return np.clip(image * strength, np.min(image), np.max(image) * 2)
        elif filter_type == 'contrast':
            mean_val = np.mean(image)
            return np.clip(mean_val + (image - mean_val) * strength, np.min(image), np.max(image))
        elif filter_type == 'gaussian':
            try:
                from scipy.ndimage import gaussian_filter
                sigma = strength * 0.8
                return gaussian_filter(image, sigma=sigma)
            except ImportError:
                logger.warning("SciPy not available, using simple smoothing")
                return simple_smooth(image, int(strength))
        elif filter_type == 'median':
            try:
                from scipy.ndimage import median_filter
                size = max(3, int(strength * 3))
                if size % 2 == 0:
                    size += 1
                return median_filter(image, size=size)
            except ImportError:
                return image
        elif filter_type == 'sobel':
            try:
                from scipy.ndimage import sobel
                return np.abs(sobel(image)) * strength
            except ImportError:
                return image
        else:
            return image
    except Exception as e:
        logger.error(f"Error applying filter {filter_type}: {e}")
        return image

def apply_enhanced_intensity_scaling(image: np.ndarray, scale_type: str) -> np.ndarray:
    """Apply intensity scaling"""
    try:
        if scale_type == 'linear':
            scaled = image
        elif scale_type == 'log':
            scaled = np.log10(np.maximum(image - np.min(image) + 1, 1))
        elif scale_type == 'sqrt':
            scaled = np.sqrt(np.maximum(image - np.min(image), 0))
        elif scale_type == 'asinh':
            median_val = np.median(image)
            mad = np.median(np.abs(image - median_val))
            scaled = np.arcsinh((image - median_val) / (3 * mad + 1e-10))
        else:
            scaled = image
        
        # Apply contrast stretch
        try:
            from skimage import exposure
            p1, p99 = np.percentile(scaled, (1, 99))
            if p99 > p1:
                stretched = exposure.rescale_intensity(scaled, in_range=(p1, p99))
                enhanced = exposure.adjust_gamma(stretched, gamma=0.8)
                logger.info(f"Applied contrast stretch and gamma correction for {scale_type} scaling")
                return enhanced
            else:
                return scaled
        except ImportError:
            p1, p99 = np.percentile(scaled, (1, 99))
            if p99 > p1:
                stretched = np.clip((scaled - p1) / (p99 - p1), 0, 1)
                enhanced = np.power(stretched, 0.8)
                return enhanced
            else:
                return scaled
                
    except Exception as e:
        logger.error(f"Error applying intensity scaling: {e}")
        return image

def simple_smooth(image, kernel_size):
    """Simple smoothing fallback"""
    if kernel_size <= 1:
        return image
        
    kernel_size = min(kernel_size, 5)
    pad_size = kernel_size // 2
    padded = np.pad(image, pad_size, mode='edge')
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):  # FIXED: was image.shape[7]
            result[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
    
    return result
