#!/usr/bin/env python3
"""
FraudLens Flask Web Interface
Alternative to Gradio for Gmail fraud detection
"""

from flask import Flask, render_template, request, jsonify
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraudlens.api.gmail_imap_integration import GmailIMAPScanner
from fraudlens.processors.text.detector import TextFraudDetector

app = Flask(__name__)

# Global state
gmail_scanner = None
text_detector = TextFraudDetector()
email_stats = {
    "total_processed": 0,
    "fraud_detected": 0,
    "last_check": "Never"
}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/connect', methods=['POST'])
def connect_gmail():
    """Connect to Gmail"""
    global gmail_scanner
    
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required"})
    
    try:
        gmail_scanner = GmailIMAPScanner(fraud_detector=text_detector)
        success = gmail_scanner.connect(email, password)
        
        if success:
            return jsonify({"success": True, "message": f"Connected to {email}"})
        else:
            return jsonify({"success": False, "message": "Connection failed"})
            
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/scan', methods=['POST'])
def scan_emails():
    """Scan emails for fraud"""
    global email_stats
    
    if not gmail_scanner or not gmail_scanner.is_connected:
        return jsonify({"success": False, "message": "Not connected to Gmail"})
    
    data = request.json
    query_type = data.get('query_type', 'Unread')
    max_emails = data.get('max_emails', 100)
    
    # Build query
    query_map = {
        'Unread': 'UNSEEN',
        'All': 'ALL',
        'With Attachments': 'ALL',  # Will filter after
        'Unknown Senders': 'ALL'    # Will filter after
    }
    
    query = query_map.get(query_type, 'UNSEEN')
    
    try:
        # Scan emails
        results = asyncio.run(
            gmail_scanner.scan_for_fraud(
                query=query,
                max_emails=max_emails
            )
        )
        
        # Update stats
        email_stats['total_processed'] += len(results)
        email_stats['last_check'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        fraud_count = sum(1 for r in results if r.get('is_fraud', False))
        email_stats['fraud_detected'] += fraud_count
        
        return jsonify({
            "success": True,
            "results": results,
            "stats": email_stats
        })
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/disconnect', methods=['POST'])
def disconnect_gmail():
    """Disconnect from Gmail"""
    global gmail_scanner
    
    if gmail_scanner:
        gmail_scanner.disconnect()
        gmail_scanner = None
    
    return jsonify({"success": True, "message": "Disconnected"})

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(email_stats)

if __name__ == '__main__':
    logger.info("Starting FraudLens Flask server...")
    
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=7863,
        debug=True
    )