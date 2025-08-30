#!/bin/bash

# FraudLens Analytics Dashboard Launcher

echo "=================================================="
echo "      FraudLens Analytics Dashboard"
echo "=================================================="
echo ""
echo "📊 Starting analytics dashboard..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
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

echo "✅ All dependencies installed"
echo ""
echo "🚀 Launching dashboard..."
echo "📍 Dashboard will be available at: http://localhost:7862"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Launch the dashboard
python3 demo/gradio_analytics_dashboard.py