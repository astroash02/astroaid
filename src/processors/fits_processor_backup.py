"""

FITS file processor with support for images, tables, and headers

Enhanced for AstroAid Dashboard with robust error handling

"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
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

def safely_extract_tuple(value, index=0, default=1920):
    """Safely extract value from potentially nested tuple structures"""
    try:
        if isinstance(value, (list, tuple)) and len(value) > index:
            result = value[index]
            while isinstance(result, (list, tuple)) and len(result) > 0:
                result = result[0]
            return int(result) if isinstance(result, (int, float)) else default
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return default
    except:
        return default

class FITSProcessor:
    """Handles FITS file processing with comprehensive HDU support"""

    def __init__(self):
        # ULTRA SAFE: max_image_size initialization
        try:
            max_size_config = config.FITS_MAX_IMAGE_SIZE
            width = safely_extract_tuple(max_size_config, 0, 1920)
            height = safely_extract_tuple(max_size_config, 1, 1080)
            self.max_image_size = (width, height)
        except Exception:
            self.max_image_size = (1920, 1080)

        try:
            self.dpi = int(config.FITS_IMAGE_DPI)
        except Exception:
            self.dpi = 100

    def process_file(self, contents: str, filename: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process uploaded FITS file

        Args:
            contents: Base64 encoded file contents
            filename: Name of the uploaded file

        Returns:
            Tuple of (HDU list, file metadata)
        """
        try:
            # Decode file contents
            content_type, content_string = contents.split(',')
            decoded_content = base64.b64decode(content_string)
            logger.info(f"Processing FITS file: {filename}")

            # Open FITS file with verification disabled to handle ASCII table issues
            with fits.open(BytesIO(decoded_content), ignore_missing_simple=True) as hdul:
                hdu_list = []
                for idx, hdu in enumerate(hdul):
                    hdu_info = self._process_hdu(hdu, idx, filename)
                    hdu_list.append(hdu_info)

                # Generate file metadata
                file_metadata = self._generate_file_metadata(hdul, filename)
                logger.info(f"Successfully processed FITS file: {len(hdu_list)} HDUs")
                return hdu_list, file_metadata

        except Exception as e:
            logger.error(f"Error processing FITS file {filename}: {e}")
            raise

    def _process_hdu(self, hdu, idx: int, filename: str) -> Dict[str, Any]:
        """Process individual HDU"""
        hdu_info = {
            'index': idx,
            'type': 'HEADER',
            'data': None,
            'image_b64': '',
            'header': {},
            'info': '',
            'table_data': pd.DataFrame() # Initialize as empty DataFrame
        }

        try:
            # Determine HDU type and extract data
            if hdu.data is not None:
                if isinstance(hdu.data, np.ndarray):
                    if hdu.data.ndim >= 2:
                        # Image HDU
                        hdu_info['type'] = 'IMAGE'
                        hdu_info['image_b64'] = self._create_image(hdu.data, f"{filename} HDU {idx}")
                        hdu_info['info'] = f"HDU {idx}: Image {hdu.data.shape}"

                    elif hdu.data.ndim == 1:
                        # 1D data (spectrum)
                        hdu_info['type'] = 'SPECTRUM'
                        hdu_info['image_b64'] = self._create_spectrum_plot(hdu.data, f"{filename} HDU {idx}")
                        hdu_info['info'] = f"HDU {idx}: Spectrum ({len(hdu.data)} points)"

                # Check for table data
                if hasattr(hdu, 'columns') and hdu.columns:
                    hdu_info['type'] = 'TABLE'
                    hdu_info['table_data'] = self._extract_table_data(hdu)
                    hdu_info['info'] = f"HDU {idx}: Table ({len(hdu.data)} rows, {len(hdu.columns)} cols)"
            else:
                hdu_info['info'] = f"HDU {idx}: Header only"

            # Extract header
            hdu_info['header'] = self._extract_header(hdu.header)

        except Exception as e:
            logger.error(f"Error processing HDU {idx}: {e}")
            hdu_info['type'] = 'ERROR'
            hdu_info['info'] = f"HDU {idx}: Error - {str(e)}"

        return hdu_info

    def _create_image(self, data: np.ndarray, title: str) -> str:
        """Create base64 encoded image from FITS data - ULTRA SAFE VERSION"""
        try:
            # Handle 3D data by taking middle slice
            if data.ndim > 2:
                if data.shape[0] > 1:
                    data = data[data.shape // 2]
                else:
                    data = data

            # FIXED: Ultra-safe shape extraction
            try:
                shape = data.shape
                if len(shape) < 2:
                    logger.error(f"Data shape insufficient for image: {shape}")
                    return ""
                h, w = int(shape[0]), int(shape[1])
            except (IndexError, TypeError) as e:
                logger.error(f"Error extracting data shape: {e}")
                return ""

            # Safe max_image_size handling (already guaranteed to be ints)
            max_h, max_w = self.max_image_size

            # Resize if too large
            if h > max_h or w > max_w:
                step_y = max(1, h // max_h)
                step_x = max(1, w // max_w)
                data = data[::step_y, ::step_x]

            # Handle NaN and infinite values
            finite_mask = np.isfinite(data)
            finite_data = data[finite_mask]

            if len(finite_data) == 0:
                logger.warning("No finite values in FITS data")
                return ""

            # Robust scaling
            vmin, vmax = np.percentile(finite_data, [2, 98])
            if vmin == vmax:
                vmin, vmax = np.nanmin(data), np.nanmax(data)
                if vmin == vmax:
                    vmax = vmin + 1

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
            im = ax.imshow(data, cmap='viridis', origin='lower',
                          vmin=vmin, vmax=vmax, interpolation='nearest')

            # FIXED: Truncate long titles and add spacing to prevent overlap
            short_title = title[:60] + '...' if len(title) > 60 else title
            ax.set_title(short_title, fontsize=11, fontweight='bold', pad=20)
            ax.set_xlabel('X (pixels)', fontsize=10)
            ax.set_ylabel('Y (pixels)', fontsize=10)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # FIXED: Prevent title overlap with proper layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9)

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight',
                       facecolor='white', edgecolor='none', dpi=self.dpi)
            plt.close(fig)

            return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            logger.error(f"Error creating FITS image: {e}")
            return ""

    def _create_spectrum_plot(self, data: np.ndarray, title: str) -> str:
        """Create spectrum plot for 1D data - ULTRA SAFE VERSION"""
        try:
            # FIXED: Handle all problematic input shapes including (1,1)
            if isinstance(data, np.ndarray):
                original_shape = data.shape
                
                if data.ndim == 0:
                    # Scalar value - create single point plot
                    data = np.array([data.item()])
                elif data.ndim > 1:
                    # Multi-dimensional array
                    if data.size == 1:
                        # Single element array like [[1]] -> [1]
                        data = np.array([data.flat])
                    elif 1 in data.shape:
                        # Arrays like (1, N) or (N, 1) -> flatten
                        data = data.flatten()
                    else:
                        # Cannot convert to 1D spectrum
                        logger.error(f"Cannot create spectrum from shape {original_shape}")
                        return ""
                
                # Ensure we have at least one data point
                if data.size == 0:
                    data = np.array([0])

            fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
            x = np.arange(len(data))
            ax.plot(x, data, 'b-', linewidth=1)
            ax.set_title(title)
            ax.set_xlabel('Channel/Pixel')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)

            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            logger.error(f"Error creating spectrum plot: {e}")
            return ""

    def _extract_table_data(self, hdu) -> pd.DataFrame:
        """Extract table data from FITS binary table HDU - ULTRA SAFE VERSION"""
        try:
            # Get column names and data (limit columns for performance)
            columns = [col.name for col in hdu.columns[:20]]  # Limit columns

            rows = []
            max_rows = min(1000, len(hdu.data))  # Limit rows for performance

            for i in range(max_rows):
                row = []
                for col_name in columns:
                    try:
                        value = hdu.data[col_name][i]

                        # FIXED: Handle ALL data types properly
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
                            # For array columns, show shape and first few values
                            if value.size > 5:
                                value = f"Array{value.shape}: [{', '.join(map(str, value.flat[:3]))}...]"
                            else:
                                value = f"Array{value.shape}: {list(value.flat)}"
                        elif isinstance(value, (np.floating, np.integer)):
                            value = float(value) if isinstance(value, np.floating) else int(value)
                        elif hasattr(value, '__len__') and not isinstance(value, str):
                            # Lists or other sequences
                            if len(value) > 5:
                                value = f"Sequence[{len(value)}]: {list(value)[:3]}..."
                            else:
                                value = str(value)
                        else:
                            # Fallback for any other type
                            value = str(value)

                        # Convert to string and limit length
                        row.append(str(value)[:100])

                    except Exception as col_error:
                        logger.debug(f"Error extracting column {col_name}: {col_error}")
                        row.append("N/A")

                rows.append(row)

            # FIXED: Ultra-safe DataFrame handling with multiple explicit checks
            try:
                if not rows:
                    logger.warning("No rows extracted from table")
                    return pd.DataFrame()
                
                df = pd.DataFrame(rows, columns=columns)
                
                # Multiple safety checks to avoid DataFrame ambiguity
                if df is None:
                    logger.warning("DataFrame is None")
                    return pd.DataFrame()
                
                # Check using len() instead of truth value
                if len(df) == 0:
                    logger.warning("DataFrame has zero length")
                    return pd.DataFrame()
                
                # Check using .empty property explicitly
                if hasattr(df, 'empty') and df.empty:
                    logger.warning("DataFrame is empty via .empty property")
                    return pd.DataFrame()
                
                # Additional safety check
                if df.shape[0] == 0:
                    logger.warning("DataFrame has zero rows via .shape")
                    return pd.DataFrame()
                
                logger.info(f"Extracted table data: {len(df)} rows, {len(df.columns)} columns")
                return df
                
            except Exception as df_error:
                logger.error(f"Error creating or validating DataFrame: {df_error}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return pd.DataFrame()

    def _extract_header(self, header) -> Dict[str, str]:
        """Extract header information"""
        try:
            header_dict = {}

            # Important astronomical keywords first
            important_keys = [
                'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'OBJECT', 'DATE-OBS', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME',
                'RA', 'DEC', 'EQUINOX', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'OBSTYPE', 'OBSERVER'
            ]

            # Add important keywords first
            for key in important_keys:
                if key in header:
                    try:
                        value = header[key]
                        # Handle different value types
                        if hasattr(value, 'strip'):
                            value = str(value).strip()
                        header_dict[key] = str(value)[:100]  # Limit length
                    except Exception as e:
                        logger.debug(f"Error processing header key {key}: {e}")
                        continue

            # Add other keywords (limited to prevent excessive data)
            added_count = 0
            max_additional_keys = 20

            for key in header.keys():
                if key not in header_dict and added_count < max_additional_keys:
                    try:
                        value = header[key]
                        if hasattr(value, 'strip'):
                            value = str(value).strip()

                        # Skip empty or very long values
                        str_value = str(value)
                        if str_value and len(str_value) < 200:
                            header_dict[key] = str_value[:100]
                            added_count += 1

                    except Exception as e:
                        logger.debug(f"Error processing header key {key}: {e}")
                        continue

            logger.info(f"Extracted {len(header_dict)} header keywords")
            return header_dict

        except Exception as e:
            logger.error(f"Error extracting header: {e}")
            return {"error": f"Header extraction failed: {str(e)}"}

    def _generate_file_metadata(self, hdul, filename: str) -> Dict[str, Any]:
        """Generate metadata for FITS file"""
        try:
            # Extract basic information
            metadata = {
                'filename': filename,
                'file_type': 'fits',
                'n_hdus': len(hdul),
                'hdu_types': [],
                'data_shapes': [],
                'primary_header': {}
            }

            # Process each HDU for metadata
            for i, hdu in enumerate(hdul):
                # HDU type
                hdu_type = type(hdu).__name__
                metadata['hdu_types'].append(hdu_type)

                # Data shape
                if hdu.data is not None:
                    metadata['data_shapes'].append(list(hdu.data.shape))
                else:
                    metadata['data_shapes'].append(None)

            # Primary header (first HDU)
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

            logger.info(f"Generated metadata for FITS file: {metadata['n_hdus']} HDUs")
            return metadata

        except Exception as e:
            logger.error(f"Error generating FITS metadata: {e}")
            return {
                'filename': filename,
                'file_type': 'fits',
                'n_hdus': 0,
                'error': str(e)
            }
