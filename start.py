#!/usr/bin/env python
"""
Startup script for AstroAid Dashboard
Ensures proper conda environment activation
"""

import sys
import os
import subprocess

def check_conda_env():
    """Check if we're in the correct conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    if conda_env != 'astroaid':
        print(f"⚠️ Warning: Not in 'astroaid' conda environment (current: {conda_env})")
        print("Please run: conda activate astroaid")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['dash', 'pandas', 'numpy', 'astropy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: conda install -c conda-forge " + ' '.join(missing_packages))
        return False
    return True

def main():
    """Main startup function"""
    print("🚀 Starting AstroAid Dashboard...")
    
    # Check environment
    if not check_conda_env():
        sys.exit(1)
    
    # Check dependencies  
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ Environment checks passed")
    print(f"🐍 Python: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Import and run the app - FIXED
    try:
        from app import app
        print("📊 Starting Dash server...")
        app.run_server(debug=True, host='127.0.0.1', port=8050)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Make sure app.py exists and is properly configured")
        sys.exit(1)

if __name__ == '__main__':
    main()
