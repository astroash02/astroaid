"""
Configuration settings for AstroAid Dashboard
Optimized for astronomical data processing
ENHANCED: Image format support with clean configuration
"""

import os

# Application settings
DEBUG = True
HOST = '127.0.0.1'
PORT = 8050

# File handling settings
MAX_FILE_SIZE_MB = 500 # Increased for astronomical data

ALLOWED_EXTENSIONS = {
    'csv': ['.csv', '.tsv', '.dat', '.txt'],
    'fits': ['.fits', '.fit', '.fts'],
    'excel': ['.xlsx', '.xls'],
    # NEW: Image format support
    'image': ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'],
    'scientific_image': ['.npy', '.npz']
}

# Data processing settings
MAX_PREVIEW_ROWS = 2000
MAX_DISPLAY_COLUMNS = 200
CHUNK_SIZE = 50000

# FITS specific settings - KEEP AS IS (don't change)
FITS_IMAGE_DPI = 100
FITS_MAX_IMAGE_SIZE = (800, 600)  # Keep your existing working config

# NEW: Image processing settings (separate from FITS)
IMAGE_MAX_SIZE = (1920, 1080)  # Clean tuple for new image processor
IMAGE_QUALITY = 85
SUPPORTED_COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'grayscale', 'hot', 'cool']

# UI settings
BRAND_NAME = "AstroAid Dashboard"
PRIMARY_COLOR = "#2E86AB"
SECONDARY_COLOR = "#A23B72"

print("✅ AstroAid configuration loaded successfully")
