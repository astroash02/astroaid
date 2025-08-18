"""
FITS file processor with support for images, tables, and headers
Enhanced for AstroAid Dashboard with robust error handling
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import base64
from io import BytesIO
import pandas as pd
from typing import Tuple, Dict, Any, List
import logging
import config
import warnings

# Suppress astropy warnings about invalid TDISP formats (harmless)
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

logger = logging.getLogger(__name__)

class FITSProcessor:
    """Handles FITS file processing with comprehensive HDU support"""
    
    def __init__(self):
        # ULTRA SAFE: max_image_size initialization
        try:
            max_size_config = getattr(config, 'FITS_MAX_IMAGE_SIZE', (1920, 1080))
            if isinstance(max_size_config, tuple) and len(max_size_config) >= 2:
                width = max_size_config[0]
                height = max_size_config[1]
                
                # If values are tuples, unwrap them
                while isinstance(width, tuple) and len(width) > 0:
                    width = width
                while isinstance(height, tuple) and len(height) > 0:
                    height = height
                
                width = int(width) if isinstance(width, (int, float)) else 1920
                height = int(height) if isinstance(height, (int, float)) else 1080
                
                self.max_image_size = (width, height)
            else:
                self.max_image_size = (1920, 1080)
        except Exception as e:
            logger.error(f"Error in max_image_size initialization: {e}")
            self.max_image_size = (1920, 1080)
        
        try:
            self.dpi = int(getattr(config, 'FITS_IMAGE_DPI', 100))
        except Exception:
            self.dpi = 100

    def process_file(self, contents: str, filename: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process uploaded FITS file
        """
        try:
            content_type, content_string = contents.split(',')
            decoded_content = base64.b64decode(content_string)
            
            logger.info(f"Processing FITS file: {filename}")
            
            with fits.open(BytesIO(decoded_content), ignore_missing_simple=True) as hdul:
                hdu_list = []
                
                for idx, hdu in enumerate(hdul):
                    hdu_info = self._process_hdu(hdu, idx, filename)
                    hdu_list.append(hdu_info)
                
                file_metadata = self._generate_file_metadata(hdul, filename)
                
            logger.info(f"Successfully processed FITS file: {len(hdu_list)} HDUs")
            return hdu_list, file_metadata
            
        except Exception as e:
            logger.error(f"Error processing FITS file {filename}: {e}")
            raise

    def _process_hdu(self, hdu, idx: int, filename: str) -> Dict[str, Any]:
        """Process individual HDU - FIXED VERSION"""
        hdu_info = {
            'index': idx,
            'type': 'HEADER',
            'data': None,
            'image_b64': '',
            'header': {},
            'info': '',
            'table_data': None
        }
        
        try:
            # CRITICAL FIX: Better detection of image data
            if hdu.data is not None:
                data_shape = hdu.data.shape
                logger.info(f"HDU {idx} data shape: {data_shape}, ndim: {hdu.data.ndim}")
                
                if isinstance(hdu.data, np.ndarray):
                    # Check for valid image data (2D or higher dimensions with meaningful size)
                    if hdu.data.ndim >= 2 and hdu.data.size > 1:
                        # Check if it's actually an image (not a degenerate array)
                        non_unity_dims = [dim for dim in data_shape if dim > 1]
                        if len(non_unity_dims) >= 2:
                            hdu_info['type'] = 'IMAGE'
                            hdu_info['image_b64'] = self._create_image(hdu.data, f"{filename} HDU {idx}")
                            hdu_info['info'] = f"HDU {idx}: Image {data_shape}"
                        else:
                            hdu_info['type'] = 'SPECTRUM'
                            hdu_info['image_b64'] = self._create_spectrum_plot(hdu.data, f"{filename} HDU {idx}")
                            hdu_info['info'] = f"HDU {idx}: Spectrum ({hdu.data.size} points)"
                    elif hdu.data.ndim == 1 and hdu.data.size > 1:
                        hdu_info['type'] = 'SPECTRUM'
                        hdu_info['image_b64'] = self._create_spectrum_plot(hdu.data, f"{filename} HDU {idx}")
                        hdu_info['info'] = f"HDU {idx}: Spectrum ({len(hdu.data)} points)"
                
                # Check for table data
                if hasattr(hdu, 'columns') and hdu.columns:
                    hdu_info['type'] = 'TABLE'
                    table_df = self._extract_table_data(hdu)
                    if not table_df.empty:
                        hdu_info['table_data'] = {
                            'data': table_df.to_dict('records'),
                            'columns': table_df.columns.tolist()
                        }
                    hdu_info['info'] = f"HDU {idx}: Table ({len(hdu.data)} rows, {len(hdu.columns)} cols)"
            else:
                hdu_info['info'] = f"HDU {idx}: Header only"
            
            hdu_info['header'] = self._extract_header(hdu.header)
            
            # Log the result
            logger.info(f"HDU {idx}: type={hdu_info['type']}, has_image={bool(hdu_info['image_b64'])}")
            
        except Exception as e:
            logger.error(f"Error processing HDU {idx}: {e}")
            hdu_info['type'] = 'ERROR'
            hdu_info['info'] = f"HDU {idx}: Error - {str(e)}"
        
        return hdu_info

    def _safe_extract_dimensions(self, data):
        """BULLETPROOF dimension extraction that handles any nested tuple scenario"""
        try:
            # Get the shape
            shape = data.shape
            logger.debug(f"Original data shape: {shape}, type: {type(shape)}")
            
            # Recursively unwrap nested tuples until we get integers
            while isinstance(shape, tuple):
                if len(shape) == 0:
                    return 1, 1  # fallback
                
                # Check if all elements are integers (we're done unwrapping)
                if all(isinstance(x, (int, np.integer)) for x in shape):
                    break
                    
                # If first element is still a tuple, unwrap it
                if isinstance(shape[0], tuple):
                    shape = shape
                else:
                    break
            
            # Now extract dimensions safely
            if len(shape) >= 2:
                height = shape
                width = shape[1]
                
                # Convert to int with ultra-safe checking
                if isinstance(height, (int, np.integer)):
                    h = int(height)
                else:
                    h = 1
                    
                if isinstance(width, (int, np.integer)):
                    w = int(width)
                else:
                    w = 1
                    
                logger.debug(f"Extracted dimensions: height={h}, width={w}")
                return h, w
            else:
                return 1, 1
                
        except Exception as e:
            logger.error(f"Error extracting dimensions: {e}")
            return 1, 1

    def _create_image(self, data: np.ndarray, title: str) -> str:
        """Create base64 encoded image from FITS data - BULLETPROOF VERSION"""
        try:
            logger.info(f"Creating image for {title}, data shape: {data.shape}, ndim: {data.ndim}")
            
            # Handle 3D data by taking middle slice
            if data.ndim > 2:
                # Find the dimensions that are > 1
                non_unity_dims = [(i, dim) for i, dim in enumerate(data.shape) if dim > 1]
                if len(non_unity_dims) >= 2:
                    # Take a slice from the first dimension if it's > 1
                    if data.shape[0] > 1:
                        data = data[data.shape // 2]
                    else:
                        data = data
                else:
                    data = data.squeeze()  # Remove dimensions of size 1
            
            # Check if we have valid 2D data
            if data.ndim < 2:
                logger.error(f"Data shape insufficient for image: {data.shape}")
                return ""
            
            # BULLETPROOF dimension extraction
            h, w = self._safe_extract_dimensions(data)
            
            if h <= 0 or w <= 0:
                logger.error(f"Invalid dimensions: height={h}, width={w}")
                return ""
            
            logger.info(f"Image dimensions: {h}x{w}")
            
            max_h, max_w = self.max_image_size
            
            # Resize if too large
            if h > max_h or w > max_w:
                step_y = max(1, h // max_h)
                step_x = max(1, w // max_w)
                data = data[::step_y, ::step_x]
                logger.info(f"Resized image from {h}x{w} to {data.shape}")
            
            # Handle NaN and infinite values
            finite_mask = np.isfinite(data)
            finite_data = data[finite_mask]
            
            if len(finite_data) == 0:
                logger.warning("No finite values in FITS data")
                return ""
            
            vmin, vmax = np.percentile(finite_data, [2, 98])
            if vmin == vmax:
                vmin, vmax = np.nanmin(data), np.nanmax(data)
                if vmin == vmax:
                    vmax = vmin + 1
            
            logger.info(f"Image value range: {vmin} to {vmax}")
            
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
            im = ax.imshow(data, cmap='viridis', origin='lower',
                          vmin=vmin, vmax=vmax, interpolation='nearest')
            
            short_title = title[:60] + '...' if len(title) > 60 else title
            ax.set_title(short_title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            base64_string = base64.b64encode(buffer.getvalue()).decode()
            logger.info(f"Successfully created image for {title}")
            return base64_string
            
        except Exception as e:
            logger.error(f"Error creating FITS image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def _create_spectrum_plot(self, data: np.ndarray, title: str) -> str:
        """Create spectrum plot for 1D data - FIXED FOR (1,1) SHAPE"""
        try:
            # FIXED: Handle various data shapes properly
            if data.ndim == 0:
                data = np.array([data.item()])
            elif data.shape == (1, 1):
                scalar_value = data.item()
                data = np.array([scalar_value])
            elif data.ndim > 1:
                if 1 in data.shape:
                    data = data.flatten()
                else:
                    logger.error(f"Cannot create spectrum from shape {data.shape}")
                    return ""
            
            # Ensure we have valid 1D data
            if data.size == 0:
                data = np.array([0])
            elif data.size == 1:
                # For single-point data, create a simple representation
                data = np.array([data.item(), data.item()])
            
            fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
            x = np.arange(len(data))
            ax.plot(x, data, 'b-', linewidth=1, marker='o' if len(data) <= 10 else None)
            ax.set_title(title[:60] + '...' if len(title) > 60 else title)
            ax.set_xlabel('Channel/Pixel')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            base64_string = base64.b64encode(buffer.getvalue()).decode()
            logger.info(f"Successfully created spectrum plot for {title}")
            return base64_string
            
        except Exception as e:
            logger.error(f"Error creating spectrum plot: {e}")
            return ""

    def _extract_table_data(self, hdu) -> pd.DataFrame:
        """Extract table data from FITS binary table HDU - COMPLETELY FIXED"""
        try:
            columns = [col.name for col in hdu.columns[:20]]
            rows = []
            max_rows = min(1000, len(hdu.data))
            
            for i in range(max_rows):
                row = []
                for col_name in columns:
                    try:
                        value = hdu.data[col_name][i]
                        
                        if value is None:
                            value = "None"
                        elif isinstance(value, tuple):
                            value = str(value)[:100]
                        elif isinstance(value, (bytes, np.bytes_)):
                            try:
                                value = value.decode('utf-8', errors='replace')
                            except:
                                value = str(value)[:100]
                        elif isinstance(value, np.ndarray):
                            if value.size > 5:
                                value = f"Array{value.shape}: [{', '.join(map(str, value.flat[:3]))}...]"
                            else:
                                value = f"Array{value.shape}: {list(value.flat)}"
                        elif isinstance(value, (np.floating, np.integer)):
                            value = float(value) if isinstance(value, np.floating) else int(value)
                        elif hasattr(value, '__len__') and not isinstance(value, str):
                            if len(value) > 5:
                                value = f"Sequence[{len(value)}]: {list(value)[:3]}..."
                            else:
                                value = str(value)
                        else:
                            value = str(value)
                        
                        row.append(str(value)[:100])
                        
                    except Exception as col_error:
                        logger.debug(f"Error extracting column {col_name}: {col_error}")
                        row.append("N/A")
                
                rows.append(row)
            
            if len(rows) == 0:
                logger.warning("No rows extracted from table")
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=columns)
            
            if hasattr(df, 'empty') and df.empty:
                logger.warning("DataFrame is empty")
                return pd.DataFrame()
            elif len(df) == 0:
                logger.warning("DataFrame has zero rows")
                return pd.DataFrame()
            else:
                logger.info(f"Extracted table data: {len(df)} rows, {len(df.columns)} columns")
                return df
                
        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return pd.DataFrame()

    def _extract_header(self, header) -> Dict[str, str]:
        """Extract header information"""
        try:
            header_dict = {}
            
            important_keys = [
                'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'OBJECT', 'DATE-OBS', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME',
                'RA', 'DEC', 'EQUINOX', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'OBSTYPE', 'OBSERVER'
            ]
            
            for key in important_keys:
                if key in header:
                    try:
                        value = header[key]
                        if hasattr(value, 'strip'):
                            value = str(value).strip()
                        header_dict[key] = str(value)[:100]
                    except Exception as e:
                        logger.debug(f"Error processing header key {key}: {e}")
                        continue
            
            added_count = 0
            max_additional_keys = 20
            
            for key in header.keys():
                if key not in header_dict and added_count < max_additional_keys:
                    try:
                        value = header[key]
                        if hasattr(value, 'strip'):
                            value = str(value).strip()
                        
                        str_value = str(value)
                        if str_value and len(str_value) < 200:
                            header_dict[key] = str_value[:100]
                            added_count += 1
                    except Exception as e:
                        logger.debug(f"Error processing header key {key}: {e}")
                        continue
            
            return header_dict
            
        except Exception as e:
            logger.error(f"Error extracting header: {e}")
            return {"error": f"Header extraction failed: {str(e)}"}

    def _generate_file_metadata(self, hdul, filename: str) -> Dict[str, Any]:
        """Generate metadata for FITS file"""
        try:
            metadata = {
                'filename': filename,
                'file_type': 'fits',
                'n_hdus': len(hdul),
                'hdu_types': [],
                'data_shapes': [],
                'primary_header': {}
            }
            
            for i, hdu in enumerate(hdul):
                hdu_type = type(hdu).__name__
                metadata['hdu_types'].append(hdu_type)
                
                if hdu.data is not None:
                    metadata['data_shapes'].append(list(hdu.data.shape))
                else:
                    metadata['data_shapes'].append(None)
            
            if len(hdul) > 0:
                try:
                    primary_header = {}
                    for key in ['TELESCOP', 'INSTRUME', 'OBJECT', 'DATE-OBS', 'EXPTIME']:
                        if key in hdul[0].header:
                            primary_header[key] = str(hdul.header[key])
                    metadata['primary_header'] = primary_header
                except Exception as e:
                    logger.debug(f"Error extracting primary header: {e}")
                    metadata['primary_header'] = {}
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating FITS metadata: {e}")
            return {
                'filename': filename,
                'file_type': 'fits',
                'n_hdus': 0,
                'error': str(e)
            }
