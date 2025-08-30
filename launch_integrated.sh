#!/bin/bash

# FraudLens Integrated Dashboard Launcher
# Complete fraud detection suite with Gmail integration

echo "=================================================="
echo "      FraudLens - Advanced Fraud Detection"
echo "=================================================="
echo ""
echo "🔍 Starting integrated fraud detection dashboard..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check and install required packages
echo "📦 Checking dependencies..."

# Core dependencies
python3 -c "import gradio" 2>/dev/null || {
    echo "Installing Gradio..."
    pip3 install gradio
}

python3 -c "import plotly" 2>/dev/null || {
    echo "Installing Plotly..."
    pip3 install plotly
}

python3 -c "import reportlab" 2>/dev/null || {
    echo "Installing ReportLab..."
    pip3 install reportlab
}

# SSL certificate handling
python3 -c "import certifi" 2>/dev/null || {
    echo "Installing Certifi for SSL certificates..."
    pip3 install certifi
}

# Computer vision dependencies
python3 -c "import cv2" 2>/dev/null || {
    echo "Installing OpenCV..."
    pip3 install opencv-python
}

# Other required packages
python3 -c "import numpy" 2>/dev/null || pip3 install numpy
python3 -c "import PIL" 2>/dev/null || pip3 install Pillow
python3 -c "import pandas" 2>/dev/null || pip3 install pandas

echo "✅ All dependencies installed"
echo ""
echo "📧 Gmail Integration Notes:"
echo "   • Use app-specific password from Google Account settings"
echo "   • Enable 2-factor authentication first"
echo "   • Generate app password at: myaccount.google.com/apppasswords"
echo ""
echo "🚀 Launching integrated dashboard..."
echo "📍 Dashboard will be available at: http://localhost:7863"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Set SSL certificate bundle for macOS
export SSL_CERT_FILE=$(python3 -m certifi)
export REQUESTS_CA_BUNDLE=$(python3 -m certifi)

# Launch the integrated dashboard
python3 demo/gradio_app_integrated.py