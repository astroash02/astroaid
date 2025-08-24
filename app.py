"""
AstroAid Dashboard - Main Application Entry Point
Enhanced with Target 2: Advanced Image Display & Processing
FIXED: Pattern matching callback issues and upload functionality
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import logging
import os
from datetime import datetime

# Configure logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('astroaid.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,  # CRITICAL: Allow dynamic components
    title="AstroAid Dashboard",
    update_title="Loading...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Advanced astronomical data analysis dashboard"},
        {"name": "author", "content": "AstroAid Team"}
    ]
)

# Expose server for deployment
server = app.server

# Main Layout
app.layout = html.Div([
    # Store components for data persistence
    dcc.Store(id='session-data'),
    dcc.Store(id='file-metadata'),
    dcc.Store(id='processed-data'),
    dcc.Store(id='fits-data-store'),
    
    # Header
    dbc.NavbarSimple(
        brand="🔭 AstroAid Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Main container
    dbc.Container([
        # Welcome screen
        html.Div(id="welcome-screen", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H2("🔭 Welcome to AstroAid Dashboard", className="text-center mb-4"),
                    html.P("Advanced astronomical data analysis with interactive image processing",
                           className="text-center text-muted mb-4"),
                    
                    # File upload area - SIMPLIFIED AND FIXED
                    dcc.Upload(
                        id='file-upload',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-2x mb-3", 
                                  style={'color': '#007bff'}),
                            html.H6("Drag & Drop Files Here", 
                                   className="text-primary mb-3"),
                            html.P("or click to browse", 
                                  className="text-muted mb-3"),
                            
                            # Simplified format info
                            html.Div([
                                html.Span("Supported: ", className="text-muted small me-2"),
                                html.Span("CSV, FITS, JPG, PNG, TIFF, NPY, NPZ", 
                                         className="text-primary small fw-bold")
                            ], className="mb-3"),
                            
                            html.P("Max size: 500MB per file", 
                                  className="text-muted small")
                        ], className="text-center p-4"),
                        style={
                            'width': '100%',
                            'height': '180px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '8px',
                            'borderColor': '#007bff',
                            'backgroundColor': '#f8f9ff',
                            'cursor': 'pointer',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'transition': 'all 0.3s ease'
                        },
                        multiple=True,
                        accept='.csv,.tsv,.txt,.dat,.fits,.fit,.fts,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.gif,.npy,.npz'
                    )
                ])
            ]),
            
            # Upload status with debug info
            html.Div(id='upload-status'),
            
            # Debug info (will be hidden in production)
            html.Div(id='debug-info', children=[
                dbc.Alert([
                    html.H6("Debug Information:", className="alert-heading"),
                    html.P("Ready to receive files. Upload a file to test the connection.", 
                          className="mb-0")
                ], color="info", className="mt-3")
            ])
        ]),
        
        # Main display area
        html.Div(
            id="main-display-area",
            style={"display": "none"}
        ),
        
        # Clear button
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-trash me-2"), "Clear All Data"],
                id='clear-uploads-btn',
                color="danger",
                outline=True,
                className="mt-4"
            )
        ], className="text-center")
        
    ], fluid=True)
])

def register_all_callbacks():
    """Register all application callbacks in correct order"""
    try:
        print("🔧 Starting callback registration...")
        
        # 1. File handling callbacks FIRST (creates components)
        from src.callbacks import file_handling
        file_handling.register_callbacks(app)
        print("✅ File handling callbacks registered")
        
        # 2. Visualization callbacks
        try:
            from src.callbacks import visualization
            visualization.register_callbacks(app)
            print("✅ Visualization callbacks registered")
        except ImportError as e:
            print(f"⚠️ Visualization callbacks not found: {e}")
        
        # 3. FITS processing callbacks
        try:
            from src.callbacks import fits_processing
            fits_processing.register_fits_processing_callbacks(app)
            print("✅ FITS processing callbacks registered")
        except ImportError as e:
            print(f"⚠️ FITS processing callbacks not found: {e}")
            
        # 4. Image processing callbacks
        try:
            from src.callbacks import image_processing
            image_processing.register_image_processing_callbacks(app)
            print("✅ Image processing callbacks registered")
        except ImportError as e:
            print(f"⚠️ Image processing callbacks not found: {e}")
        
        print("🔧 All callback registration attempts completed")
        
    except Exception as e:
        print(f"❌ Error registering callbacks: {e}")
        logger.error(f"Callback registration failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")

def initialize_app():
    """Initialize application components and processors"""
    try:
        # Check processor availability first
        print("🔍 Checking processor availability...")
        
        processor_status = []
        try:
            from src.processors.csv_processor import CSVProcessor
            processor_status.append("✅ CSV Processor available")
        except Exception as e:
            processor_status.append(f"❌ CSV Processor failed: {e}")
        
        try:
            from src.processors.fits_processor import FITSProcessor
            processor_status.append("✅ FITS Processor available")
        except Exception as e:
            processor_status.append(f"❌ FITS Processor failed: {e}")
        
        try:
            from src.processors.image_processor import ImageProcessor
            processor_status.append("✅ Image Processor available")
        except Exception as e:
            processor_status.append(f"❌ Image Processor failed: {e}")
        
        for status in processor_status:
            print(status)
        
        # Register all callbacks in correct order
        register_all_callbacks()
        
        # Log startup information
        logger.info("AstroAid Dashboard initialized successfully")
        logger.info(f"Startup time: {datetime.now()}")
        logger.info("Supported file formats:")
        logger.info(" - CSV/TSV data files")
        logger.info(" - FITS astronomical images")
        logger.info(" - Standard images (JPEG, PNG, TIFF, BMP)")
        logger.info(" - Scientific images (NPY, NPZ)")
        
        print("\n🚀 AstroAid Dashboard Ready!")
        print("📊 Features Available:")
        print(" ✅ CSV/TSV data analysis")
        print(" ✅ FITS astronomical image processing")
        print(" ✅ Interactive image filters and colormaps")
        print(" ✅ Real-time image enhancement")
        print(" ✅ Multi-format support")
        print(f"\n📡 Access your dashboard at: http://127.0.0.1:8050")  # DEFAULT PORT
        print("\n🔍 Upload Debug: Watch console for upload messages starting with 🔍")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize app: {e}")
        logger.error(f"App initialization failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

# Application startup
if __name__ == '__main__':
    if initialize_app():
        try:
            print("\n🌟 Starting server...")
            # DEFAULT SERVER CONFIGURATION
            app.run_server(debug=True)  # Back to default: localhost:8050, debug enabled
        except Exception as e:
            print(f"❌ Server startup failed: {e}")
            logger.error(f"Server startup failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
    else:
        print("❌ Application failed to initialize. Check logs for details.")
