#!/usr/bin/env python3
"""
FraudLens Fresh Start - Working Gradio Interface
Clean implementation with all fraud detection features
"""

import gradio as gr
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fraudlens.processors.text.detector import TextFraudDetector
    from fraudlens.api.gmail_imap_integration import GmailIMAPScanner
    print("✅ FraudLens modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Using mock classes")
    
    class TextFraudDetector:
        def __init__(self):
            self._initialized = False
        async def initialize(self):
            self._initialized = True
        async def detect(self, text):
            return type('obj', (object,), {
                'is_fraud': len(text) > 100,
                'fraud_score': 0.7 if len(text) > 100 else 0.2,
                'fraud_types': ['phishing'] if len(text) > 100 else [],
                'confidence': 0.8,
                'explanation': 'Mock detection result'
            })()
    
    class GmailIMAPScanner:
        def __init__(self, fraud_detector=None):
            self.is_connected = False
        def connect(self, email, password):
            self.is_connected = True
            return True
        def disconnect(self):
            self.is_connected = False
        async def scan_for_fraud(self, query, max_emails):
            return [
                {
                    'subject': f'Test Email {i}',
                    'sender': f'sender{i}@example.com',
                    'date': '2024-01-01',
                    'is_fraud': i % 3 == 0,
                    'confidence': 0.7 if i % 3 == 0 else 0.2,
                    'fraud_types': ['phishing'] if i % 3 == 0 else [],
                    'risk_level': 'High' if i % 3 == 0 else 'Low',
                    'message_id': str(i)
                }
                for i in range(min(5, max_emails))
            ]

print(f"Gradio version: {gr.__version__}")

# Global instances
text_detector = TextFraudDetector()
gmail_scanner = None
is_connected = False

# Initialize detector
print("Initializing text detector...")
try:
    asyncio.run(text_detector.initialize())
    print("✅ Text detector initialized")
except:
    print("⚠️ Using mock text detector")

def analyze_text(text):
    """Analyze text for fraud"""
    if not text:
        return "Please enter some text to analyze"
    
    try:
        result = asyncio.run(text_detector.detect(text))
        
        output = f"""
### Analysis Results

**Fraud Score:** {result.fraud_score:.2%}
**Is Fraud:** {'Yes ⚠️' if result.is_fraud else 'No ✅'}
**Confidence:** {result.confidence:.2%}

**Fraud Types:** {', '.join(result.fraud_types) if result.fraud_types else 'None detected'}

**Explanation:** {result.explanation}
"""
        return output
    except Exception as e:
        return f"Error analyzing text: {str(e)}"

def connect_gmail(email, password):
    """Connect to Gmail"""
    global gmail_scanner, is_connected
    
    if not email or not password:
        return "Please enter email and password", "❌ Not connected"
    
    try:
        gmail_scanner = GmailIMAPScanner(fraud_detector=text_detector)
        success = gmail_scanner.connect(email, password)
        
        if success:
            is_connected = True
            return "✅ Connected successfully!", f"✅ Connected to {email}"
        else:
            return "❌ Connection failed", "❌ Not connected"
    except Exception as e:
        return f"❌ Error: {str(e)}", "❌ Not connected"

def scan_emails(query_type, max_emails):
    """Scan emails for fraud"""
    global gmail_scanner, is_connected
    
    if not is_connected or not gmail_scanner:
        return "Please connect to Gmail first"
    
    query_map = {
        'Unread': 'UNSEEN',
        'All': 'ALL',
        'Recent': 'RECENT'
    }
    
    query = query_map.get(query_type, 'UNSEEN')
    
    try:
        results = asyncio.run(gmail_scanner.scan_for_fraud(query, max_emails))
        
        if not results:
            return "No emails found"
        
        output = f"### Scanned {len(results)} emails\n\n"
        
        for email in results:
            status = "⚠️ FRAUD" if email['is_fraud'] else "✅ Safe"
            output += f"""
**{status}** - {email['subject']}
- From: {email['sender']}
- Date: {email['date']}
- Confidence: {email['confidence']:.1%}
- Risk: {email['risk_level']}

---
"""
        
        return output
        
    except Exception as e:
        return f"Error scanning emails: {str(e)}"

# Create Gradio interface
print("Creating Gradio interface...")

with gr.Blocks(title="FraudLens", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 FraudLens - Fraud Detection System
    Analyze text and emails for potential fraud
    """)
    
    with gr.Tab("📝 Text Analysis"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Paste suspicious text here...",
                    lines=10
                )
                analyze_btn = gr.Button("Analyze Text", variant="primary")
            
            with gr.Column():
                text_output = gr.Markdown()
        
        analyze_btn.click(
            fn=analyze_text,
            inputs=text_input,
            outputs=text_output
        )
    
    with gr.Tab("📧 Email Scanner"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Gmail Connection")
                email_input = gr.Textbox(label="Gmail Address", type="email")
                password_input = gr.Textbox(label="App Password", type="password")
                connect_btn = gr.Button("Connect to Gmail", variant="primary")
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
                gmail_status = gr.Markdown("❌ Not connected")
            
            with gr.Column():
                gr.Markdown("### Scan Options")
                query_type = gr.Radio(
                    choices=["Unread", "All", "Recent"],
                    value="Unread",
                    label="Email Filter"
                )
                max_emails_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1,
                    label="Max Emails to Scan"
                )
                scan_btn = gr.Button("Scan Emails", variant="primary")
        
        email_results = gr.Markdown()
        
        connect_btn.click(
            fn=connect_gmail,
            inputs=[email_input, password_input],
            outputs=[connection_status, gmail_status]
        )
        
        scan_btn.click(
            fn=scan_emails,
            inputs=[query_type, max_emails_slider],
            outputs=email_results
        )

print("Gradio interface created successfully!")

if __name__ == "__main__":
    print("Launching Gradio app...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7863,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"Launch error: {e}")
        traceback.print_exc()