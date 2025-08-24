"""
Enhanced FITS Processor with World Coordinate System (WCS) Support
Builds on working Target #2 - preserves all existing functionality
"""

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO
import pandas as pd
from typing import Tuple, Dict, Any, List
import logging
import config
import warnings

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

logger = logging.getLogger(__name__)

class FITSProcessor:
    """Enhanced FITS processor with WCS support - preserves all Target #2 functionality"""

    def __init__(self):
        try:
            max_size_config = getattr(config, 'FITS_MAX_IMAGE_SIZE', (1920, 1080))
            if isinstance(max_size_config, tuple) and len(max_size_config) >= 2:
                width = max_size_config[0]
                height = max_size_config[1]
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
        """Process uploaded FITS file - PRESERVES Target #2 functionality"""
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
        """Process individual HDU with optional WCS support"""
        hdu_info = {
            'index': idx,
            'type': 'HEADER',
            'data': None,
            'image_b64': '',
            'raw_image_data': None,
            'wcs_info': None,  # NEW: Optional WCS info
            'header': {},
            'info': '',
            'table_data': None
        }

        try:
            if hdu.data is not None:
                data_shape = hdu.data.shape
                logger.info(f"HDU {idx} data shape: {data_shape}, ndim: {hdu.data.ndim}")

                # Try to extract WCS info (optional - won't break if it fails)
                try:
                    hdu_info['wcs_info'] = self._extract_wcs_info(hdu)
                except Exception as wcs_error:
                    logger.debug(f"No WCS available for HDU {idx}: {wcs_error}")
                    hdu_info['wcs_info'] = {'wcs_available': False, 'reason': str(wcs_error)}

                if isinstance(hdu.data, np.ndarray):
                    if hdu.data.ndim >= 2 and hdu.data.size > 1:
                        non_unity_dims = [dim for dim in data_shape if dim > 1]
                        if len(non_unity_dims) >= 2:
                            hdu_info['type'] = 'IMAGE'
                            hdu_info['raw_image_data'] = self.get_raw_image_data_safe(hdu)
                            hdu_info['image_b64'] = self._create_image(hdu.data, f"{filename} HDU {idx}")
                            
                            # Enhanced info with optional WCS details
                            wcs_info = hdu_info.get('wcs_info', {})
                            if wcs_info.get('wcs_available'):
                                pixel_scale = wcs_info.get('pixel_scale', {})
                                scale_info = f", {pixel_scale.get('average_arcsec', 0):.2f}\"/px" if pixel_scale else ""
                                projection_info = f", {wcs_info.get('projection', 'Unknown')}" if wcs_info.get('projection') else ""
                                hdu_info['info'] = f"HDU {idx}: Image {data_shape}{scale_info}{projection_info}"
                            else:
                                hdu_info['info'] = f"HDU {idx}: Image {data_shape}"
                            
                    elif hdu.data.ndim == 1 and hdu.data.size > 1:
                        hdu_info['type'] = 'SPECTRUM'
                        hdu_info['image_b64'] = self._create_spectrum_plot(hdu.data, f"{filename} HDU {idx}")
                        hdu_info['info'] = f"HDU {idx}: Spectrum ({len(hdu.data)} points)"

                if hasattr(hdu, 'columns') and hdu.columns:
                    hdu_info['type'] = 'TABLE'
                    table_df = self._extract_table_data(hdu)
                    if not table_df.empty:
                        hdu_info['table_data'] = {
                            'data': table_df.to_dict('records'),
                            'columns': table_df.columns.tolist()
                        }
            else:
                hdu_info['info'] = f"HDU {idx}: Header only"

            hdu_info['header'] = self._extract_header(hdu.header)
            has_wcs = hdu_info.get('wcs_info', {}).get('wcs_available', False)
            logger.info(f"HDU {idx}: type={hdu_info['type']}, has_raw_data={hdu_info['raw_image_data'] is not None}, WCS={has_wcs}")

        except Exception as e:
            logger.error(f"Error processing HDU {idx}: {e}")
            hdu_info['type'] = 'ERROR'
            hdu_info['info'] = f"HDU {idx}: Error - {str(e)}"
            hdu_info['raw_image_data'] = self._create_fallback_data()

        return hdu_info

    def _extract_wcs_info(self, hdu):
        """Extract WCS information safely - NEW functionality"""
        try:
            wcs = WCS(hdu.header)
            
            if wcs.has_celestial:
                coordinate_system = wcs.wcs.ctype.tolist() if hasattr(wcs.wcs, 'ctype') else ['Unknown', 'Unknown']
                pixel_scale_info = self._get_pixel_scale_info(wcs)
                projection_info = self._get_projection_info(wcs)
                
                return {
                    'wcs_available': True,
                    'coordinate_system': coordinate_system,
                    'reference_pixel': wcs.wcs.crpix.tolist() if hasattr(wcs.wcs, 'crpix') else None,
                    'reference_coordinate': wcs.wcs.crval.tolist() if hasattr(wcs.wcs, 'crval') else None,
                    'pixel_scale': pixel_scale_info,
                    'projection': projection_info.get('projection_code', 'Unknown'),
                    'projection_name': projection_info.get('projection_name', 'Unknown'),
                    'equinox': getattr(wcs.wcs, 'equinox', None),
                    'radesys': getattr(wcs.wcs, 'radesys', None)
                }
            else:
                return {
                    'wcs_available': False, 
                    'reason': 'No celestial coordinates found'
                }
                
        except Exception as e:
            return {
                'wcs_available': False, 
                'reason': f'WCS parsing error: {str(e)}'
            }

    def _get_pixel_scale_info(self, wcs):
        """Get pixel scale information"""
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            pixel_scales = proj_plane_pixel_scales(wcs)
            return {
                'x_scale_arcsec': pixel_scales[0] * 3600,
                'y_scale_arcsec': pixel_scales[11] * 3600,
                'average_arcsec': np.mean(pixel_scales) * 3600,
                'x_scale_deg': pixel_scales,
                'y_scale_deg': pixel_scales[11]
            }
        except Exception as e:
            logger.debug(f"Error calculating pixel scale: {e}")
            return None

    def _get_projection_info(self, wcs):
        """Extract projection information"""
        try:
            ctype = wcs.wcs.ctype
            projection_code = 'Unknown'
            
            if len(ctype) > 0:
                if '---' in ctype:
                    projection_code = ctype.split('---')[11]
                else:
                    projection_code = ctype[-3:] if len(ctype) >= 3 else 'UNK'
            
            projection_names = {
                'TAN': 'Gnomonic (Tangent Plane)',
                'SIN': 'Orthographic',
                'ARC': 'Zenithal Equidistant',
                'STG': 'Stereographic',
                'CAR': 'Plate Carrée'
            }
            
            projection_name = projection_names.get(projection_code, f'Unknown ({projection_code})')
            
            return {
                'projection_code': projection_code,
                'projection_name': projection_name
            }
        except Exception as e:
            logger.debug(f"Error calculating pixel scale: {e}")
            return {'projection_code': 'Unknown', 'projection_name': 'Unknown'}

    def get_raw_image_data_safe(self, hdu) -> List[List[float]]:
        """PRESERVES Target #2 functionality with enhanced resolution"""
        try:
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = hdu.data.copy()
                
                # Handle 3D FITS cubes - WORKING from Target #2
                if data.ndim > 2:
                    first_dim_size = int(data.shape[0])
                    middle_slice = first_dim_size // 2
                    data = data[middle_slice]
                    logger.info(f"Extracted slice {middle_slice} from 3D FITS cube")
                
                # Handle NaN values - WORKING from Target #2
                finite_mask = np.isfinite(data)
                if np.any(finite_mask):
                    median_val = np.nanmedian(data[finite_mask])
                    data = np.where(finite_mask, data, median_val)
                
                # Convert to safe float64 and downsize - ENHANCED resolution
                data = data.astype(np.float64)
                height, width = int(data.shape[0]), int(data.shape[1])
                max_size = 1000  # ENHANCED from Target #2 (was 100)
                
                if height > max_size or width > max_size:
                    step_y = max(1, height // max_size)
                    step_x = max(1, width // max_size)
                    data = data[::step_y, ::step_x]
                    logger.info(f"Downsampled FITS data from {height}x{width} to {data.shape}")
                
                # Convert to list safely - WORKING from Target #2
                result = data.tolist()
                actual_size = f"{len(result)}x{len(result[0])}" if result else "0x0"
                logger.info(f"Successfully extracted REAL FITS data: {actual_size} pixels")
                return result
                
            else:
                logger.warning("No valid FITS data, using fallback")
                return self._create_fallback_data()
                
        except Exception as e:
            logger.error(f"Error extracting FITS data: {e}")
            logger.warning("Using fallback data due to extraction error")
            return self._create_fallback_data()

    def _create_fallback_data(self) -> List[List[float]]:
        """PRESERVES Target #2 fallback functionality"""
        size = 10
        return [[float(i + j) for j in range(size)] for i in range(size)]

    def _create_image(self, data: np.ndarray, title: str) -> str:
        """PRESERVES Target #2 image creation with enhancements"""
        try:
            logger.info(f"Creating image for {title}, data shape: {data.shape}, ndim: {data.ndim}")

            if data.ndim > 2:
                first_dim_size = int(data.shape[0])
                middle_slice = first_dim_size // 2
                data = data[middle_slice]

            if data.ndim < 2:
                logger.error(f"Data shape insufficient for image: {data.shape}")
                return ""

            max_h, max_w = self.max_image_size
            h, w = data.shape
            if h > max_h or w > max_w:
                step_y = max(1, h // max_h)
                step_x = max(1, w // max_w)
                data = data[::step_y, ::step_x]
                logger.info(f"Resized image from {h}x{w} to {data.shape}")

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

            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
            im = ax.imshow(data, cmap='viridis', origin='lower',
                          vmin=vmin, vmax=vmax, interpolation='nearest')

            short_title = title[:60] + '...' if len(title) > 60 else title
            ax.set_title(short_title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            base64_string = base64.b64encode(buffer.getvalue()).decode()
            logger.info(f"Successfully created image for {title}")
            return base64_string

        except Exception as e:
            logger.error(f"Error creating FITS image: {e}")
            return ""

    def _create_spectrum_plot(self, data: np.ndarray, title: str) -> str:
        """PRESERVES Target #2 spectrum plotting"""
        try:
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

            if data.size == 0:
                data = np.array([0])
            elif data.size == 1:
                data = np.array([data.item(), data.item()])

            fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
            x = np.arange(len(data))
            ax.plot(x, data, 'b-', linewidth=1)

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
        """PRESERVES Target #2 table extraction"""
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
                        else:
                            value = str(value)

                        row.append(str(value)[:100])
                    except Exception:
                        row.append("N/A")

                rows.append(row)

            if len(rows) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=columns)
            logger.info(f"Extracted table data: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return pd.DataFrame()

    def _extract_header(self, header) -> Dict[str, str]:
        """Enhanced header extraction with WCS keywords"""
        try:
            header_dict = {}
            important_keys = [
                'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'OBJECT', 'DATE-OBS', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME',
                'RA', 'DEC', 'EQUINOX', 'RADESYS',
                'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 
                'CDELT1', 'CDELT2', 'OBSTYPE', 'OBSERVER'
            ]

            for key in important_keys:
                if key in header:
                    try:
                        value = header[key]
                        if hasattr(value, 'strip'):
                            value = str(value).strip()
                        header_dict[key] = str(value)[:100]
                    except Exception:
                        continue

            return header_dict

        except Exception as e:
            logger.error(f"Error extracting header: {e}")
            return {"error": f"Header extraction failed: {str(e)}"}

    def _generate_file_metadata(self, hdul, filename: str) -> Dict[str, Any]:
        """Enhanced metadata generation with WCS info"""
        try:
            metadata = {
                'filename': filename,
                'file_type': 'fits',
                'n_hdus': len(hdul),
                'hdu_types': [],
                'data_shapes': [],
                'wcs_available': [],
                'primary_header': {}
            }

            for i, hdu in enumerate(hdul):
                hdu_type = type(hdu).__name__
                metadata['hdu_types'].append(hdu_type)

                if hdu.data is not None:
                    metadata['data_shapes'].append(list(hdu.data.shape))
                    # Check for WCS in each HDU
                    try:
                        wcs = WCS(hdu.header)
                        metadata['wcs_available'].append(wcs.has_celestial)
                    except:
                        metadata['wcs_available'].append(False)
                else:
                    metadata['data_shapes'].append(None)
                    metadata['wcs_available'].append(False)

            if len(hdul) > 0:
                try:
                    primary_header = {}
                    for key in ['TELESCOP', 'INSTRUME', 'OBJECT', 'DATE-OBS', 'EXPTIME', 'CTYPE1', 'CTYPE2']:
                        if key in hdul[0].header:
                            primary_header[key] = str(hdul.header[key])
                    metadata['primary_header'] = primary_header
                except Exception:
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
