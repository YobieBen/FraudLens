#!/bin/bash

# FraudLens Analytics Dashboard Launcher
echo "=========================================="
echo "    FraudLens Analytics Dashboard"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Installing Streamlit..."
    pip3 install streamlit --quiet
fi

# Check if plotly is installed
if ! python3 -c "import plotly" &> /dev/null 2>&1; then
    echo "Installing Plotly..."
    pip3 install plotly --quiet
fi

# Check if reportlab is installed
if ! python3 -c "import reportlab" &> /dev/null 2>&1; then
    echo "Installing ReportLab for PDF export..."
    pip3 install reportlab --quiet
fi

# Launch the dashboard
echo "Launching FraudLens Analytics Dashboard..."
echo ""
echo "📊 Dashboard will open at: http://localhost:7864"
echo ""
echo "Features available:"
echo "  ✓ Real-time fraud trends analysis"
echo "  ✓ Fraud type distribution charts"
echo "  ✓ Email fraud pattern heatmaps"
echo "  ✓ Geographic fraud distribution"
echo "  ✓ Export to PDF and CSV"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run demo/fraud_analytics_dashboard.py --server.port 7864