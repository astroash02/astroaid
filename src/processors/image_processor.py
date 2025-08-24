"""
BULLETPROOF image processor with safe PIL handling for all formats
"""

from PIL import Image, ImageEnhance
import numpy as np
import base64
from io import BytesIO
import cv2
import logging
import config
from typing import Tuple, Dict, Any, Optional
import traceback

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process image formats with bulletproof error handling"""
    
    def __init__(self):
        self.max_size = getattr(config, 'IMAGE_MAX_SIZE', (1920, 1080))
        self.quality = getattr(config, 'IMAGE_QUALITY', 85)
        self.supported_formats = {
            'standard': ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'],
            'scientific': ['.npy', '.npz']
        }
    
    def process_file(self, contents: str, filename: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process uploaded image file with comprehensive error handling"""
        try:
            content_type, content_string = contents.split(',')
            decoded_content = base64.b64decode(content_string)
            
            file_ext = filename.lower().split('.')[-1]
            format_type = self._detect_format(file_ext)
            
            logger.info(f"Processing {filename}: format={file_ext}, type={format_type}")
            
            if format_type == 'standard':
                return self._process_standard_image(decoded_content, filename)
            elif format_type == 'scientific':
                return self._process_scientific_image(decoded_content, filename)
            else:
                raise ValueError(f"Unsupported image format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _detect_format(self, extension: str) -> str:
        """Detect image format category"""
        for format_type, extensions in self.supported_formats.items():
            if f'.{extension}' in extensions:
                return format_type
        return 'unknown'
    
    def _process_standard_image(self, image_data: bytes, filename: str) -> Tuple[Dict, Dict]:
        """Process standard image formats with BULLETPROOF PIL handling"""
        try:
            # CRITICAL: Safe PIL image opening
            image = Image.open(BytesIO(image_data))
            logger.info(f"PIL image opened: mode={getattr(image, 'mode', 'unknown')}")
            
            # BULLETPROOF: Validate image properties
            if not hasattr(image, 'size'):
                raise ValueError("PIL image missing size attribute")
            
            size = getattr(image, 'size', None)
            if not isinstance(size, tuple) or len(size) != 2:
                raise ValueError(f"Invalid image size: {size}")
            
            width, height = size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")
            
            logger.info(f"Image size validated: {width}x{height}")
            
            # BULLETPROOF: Mode conversion
            original_mode = getattr(image, 'mode', 'unknown')
            logger.info(f"Original mode: {original_mode}")
            
            if original_mode in ('RGBA', 'LA'):
                # Handle transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                if original_mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image, mask=image.split()[-1] if len(image.split()) > 1 else None)
                image = background
            elif original_mode == 'P':
                # Handle palette mode
                image = image.convert('RGB')
            elif original_mode not in ('RGB', 'L'):
                # Convert any other mode to RGB
                image = image.convert('RGB')
            elif original_mode == 'L':
                # Convert grayscale to RGB
                image = image.convert('RGB')
            
            logger.info(f"Final mode: {getattr(image, 'mode', 'unknown')}")
            
            # BULLETPROOF: Resize check
            max_width, max_height = self.max_size
            current_width, current_height = image.size
            
            if current_width > max_width or current_height > max_height:
                logger.info(f"Resizing from {current_width}x{current_height} to max {max_width}x{max_height}")
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized to: {image.size}")
            
            # BULLETPROOF: Convert to numpy array
            try:
                image_array = np.array(image, dtype=np.uint8)
                logger.info(f"Numpy array created: shape={image_array.shape}, dtype={image_array.dtype}")
            except Exception as e:
                logger.error(f"Error converting to numpy: {e}")
                raise ValueError(f"Failed to convert image to numpy array: {e}")
            
            # Validate array
            if image_array.size == 0:
                raise ValueError("Empty image array")
            
            if len(image_array.shape) not in [2, 3]:
                raise ValueError(f"Invalid array shape: {image_array.shape}")
            
            # Create display image
            display_image = self._create_display_image(image_array, filename)
            if not display_image:
                raise ValueError("Failed to create display image")
            
            # Prepare data structure
            image_data = {
                'type': 'IMAGE',
                'original_image_b64': display_image,
                'numpy_data': image_array.tolist(),
                'current_filter': 'none',
                'current_colormap': 'original',
                'filename': filename
            }
            
            metadata = {
                'filename': filename,
                'file_type': 'image',
                'format': getattr(image, 'format', 'Unknown'),
                'size': image.size,
                'mode': image.mode,
                'original_mode': original_mode,
                'channels': len(image_array.shape),
                'dimensions': image_array.shape,
                'data_type': str(image_array.dtype),
                'color_stats': self._calculate_color_stats(image_array)
            }
            
            logger.info(f"Successfully processed {filename}")
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Error processing standard image {filename}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _process_scientific_image(self, image_data: bytes, filename: str) -> Tuple[Dict, Dict]:
        """Process scientific formats with safe handling"""
        try:
            if filename.endswith('.npy'):
                array = np.load(BytesIO(image_data))
            elif filename.endswith('.npz'):
                npz_file = np.load(BytesIO(image_data))
                array = npz_file[list(npz_file.files)[0]]
            else:
                raise ValueError(f"Unsupported scientific format")
            
            logger.info(f"Numpy array loaded: shape={array.shape}, dtype={array.dtype}")
            
            # Normalize to uint8
            normalized_array = self._safe_normalize_array(array)
            
            # Convert to RGB if needed
            if normalized_array.ndim == 2:
                display_array = np.stack([normalized_array, normalized_array, normalized_array], axis=2)
            elif normalized_array.ndim == 3 and normalized_array.shape[2] in [3, 4]:
                display_array = normalized_array[:,:,:3] if normalized_array.shape[1] == 4 else normalized_array
            else:
                raise ValueError(f"Unsupported array shape: {array.shape}")
            
            display_image = self._create_display_image(display_array, filename)
            
            image_data = {
                'type': 'SCIENTIFIC_IMAGE',
                'original_image_b64': display_image,
                'numpy_data': display_array.tolist(),
                'current_filter': 'none',
                'current_colormap': 'viridis',
                'filename': filename
            }
            
            metadata = {
                'filename': filename,
                'file_type': 'scientific_image',
                'format': 'NumPy Array',
                'dimensions': array.shape,
                'data_type': str(array.dtype),
                'min_value': float(np.min(array)),
                'max_value': float(np.max(array)),
                'mean_value': float(np.mean(array)),
                'std_value': float(np.std(array))
            }
            
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Error processing scientific image: {e}")
            raise

    def _safe_normalize_array(self, array: np.ndarray) -> np.ndarray:
        """Bulletproof array normalization"""
        try:
            if array.dtype == np.uint8:
                return array
            
            # Convert to float for operations
            array = array.astype(np.float64)
            
            array_min = np.min(array)
            array_max = np.max(array)
            
            if array_max > array_min:
                normalized = ((array - array_min) / (array_max - array_min) * 255.0)
            else:
                normalized = np.zeros_like(array, dtype=np.float64)
            
            return np.clip(normalized, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error normalizing array: {e}")
            return np.zeros(array.shape[:2] + (3,), dtype=np.uint8)

    def apply_advanced_filter(self, image_array: np.ndarray, filter_type: str, strength: float = 1.0) -> np.ndarray:
        """Apply filters with enhanced visibility"""
        try:
            # Ensure proper input
            if isinstance(image_array, list):
                image_array = np.array(image_array)
            
            if image_array.dtype != np.uint8:
                image_array = self._safe_normalize_array(image_array)
            
            # Ensure 3D array
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array, image_array, image_array], axis=2)
            
            if filter_type == 'gaussian':
                kernel_size = max(5, int(strength * 12))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), strength * 3)
                
            elif filter_type == 'median':
                kernel_size = max(3, int(strength * 10))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return cv2.medianBlur(image_array, kernel_size)
                
            elif filter_type == 'sobel':
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                sobel = np.clip(sobel * strength * 3, 0, 255).astype(np.uint8)
                return np.stack([sobel, sobel, sobel], axis=2)
                
            elif filter_type == 'unsharp':
                blurred = cv2.GaussianBlur(image_array, (9, 9), 0)
                unsharp = cv2.addWeighted(image_array, 1 + strength, blurred, -strength, 0)
                return np.clip(unsharp, 0, 255).astype(np.uint8)
                
            elif filter_type == 'brightness':
                return np.clip(image_array * (0.3 + strength * 1.4), 0, 255).astype(np.uint8)
                
            elif filter_type == 'contrast':
                return np.clip(128 + (image_array - 128) * (1 + strength), 0, 255).astype(np.uint8)
                
            else:
                return image_array
                
        except Exception as e:
            logger.error(f"Error applying filter {filter_type}: {e}")
            return image_array

    def apply_enhanced_colormap(self, image_array: np.ndarray, colormap: str) -> np.ndarray:
        """Apply colormaps with bulletproof handling"""
        try:
            if colormap == 'original':
                return image_array
            
            # Ensure proper input
            if isinstance(image_array, list):
                image_array = np.array(image_array)
                
            if image_array.dtype != np.uint8:
                image_array = self._safe_normalize_array(image_array)
            
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
                
            if gray.dtype != np.uint8:
                gray = self._safe_normalize_array(gray.astype(np.float64))
            
            # Apply colormap
            colormap_dict = {
                'hot': cv2.COLORMAP_HOT,
                'cool': cv2.COLORMAP_COOL,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'plasma': cv2.COLORMAP_PLASMA,
                'inferno': cv2.COLORMAP_INFERNO,
                'magma': cv2.COLORMAP_MAGMA,
                'grayscale': None
            }
            
            if colormap == 'grayscale':
                return np.stack([gray, gray, gray], axis=2)
            elif colormap in colormap_dict:
                colored = cv2.applyColorMap(gray, colormap_dict[colormap])
                return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            else:
                return image_array
                
        except Exception as e:
            logger.error(f"Error applying colormap {colormap}: {e}")
            return image_array

    # Legacy compatibility
    def apply_filter(self, image_array: np.ndarray, filter_type: str, **kwargs) -> np.ndarray:
        strength = kwargs.get('strength', 1.0)
        return self.apply_advanced_filter(image_array, filter_type, strength)
    
    def apply_colormap(self, image_array: np.ndarray, colormap: str) -> np.ndarray:
        return self.apply_enhanced_colormap(image_array, colormap)
    
    def _create_display_image(self, image_array: np.ndarray, title: str) -> str:
        """Create base64 image with error handling"""
        try:
            if image_array.dtype != np.uint8:
                image_array = self._safe_normalize_array(image_array)
            
            image = Image.fromarray(image_array)
            buffer = BytesIO()
            image.save(buffer, format='PNG', quality=self.quality)
            base64_string = base64.b64encode(buffer.getvalue()).decode()
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Error creating display image: {e}")
            return ""
    
    def _calculate_color_stats(self, image_array: np.ndarray) -> Dict[str, float]:
        """Calculate color statistics safely"""
        try:
            if len(image_array.shape) == 3:
                return {
                    'mean_r': float(np.mean(image_array[:, :, 0])),
                    'mean_g': float(np.mean(image_array[:, :, 1])),
                    'mean_b': float(np.mean(image_array[:, :, 2])),
                    'brightness': float(np.mean(image_array))
                }
            else:
                return {'brightness': float(np.mean(image_array))}
        except:
            return {}
