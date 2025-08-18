"""
Configuration settings for AstroAid Dashboard
Optimized for astronomical data processing
"""

import os

# Application settings
DEBUG = True
HOST = '127.0.0.1'
PORT = 8050

# File handling settings
MAX_FILE_SIZE_MB = 500  # Increased for astronomical data
ALLOWED_EXTENSIONS = {
    'csv': ['.csv', '.tsv', '.dat', '.txt'],
    'fits': ['.fits', '.fit', '.fts'],
    'excel': ['.xlsx', '.xls']
}

# Data processing settings
MAX_PREVIEW_ROWS = 2000
MAX_DISPLAY_COLUMNS = 200
CHUNK_SIZE = 50000

# FITS specific settings
FITS_IMAGE_DPI = 100
FITS_MAX_IMAGE_SIZE = (800, 600)

# UI settings
BRAND_NAME = "AstroAid Dashboard"
PRIMARY_COLOR = "#2E86AB"
SECONDARY_COLOR = "#A23B72"

print("✅ AstroAid configuration loaded successfully")
