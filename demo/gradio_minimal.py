#!/usr/bin/env python3
"""
Minimal Gradio App - No FraudLens imports
Pure Gradio interface for testing
"""

import gradio as gr
from datetime import datetime

print(f"Starting minimal Gradio app at {datetime.now()}")
print(f"Gradio version: {gr.__version__}")

def analyze_text(text):
    """Simple text analysis"""
    if not text:
        return "Please enter some text to analyze"
    
    return f"""
### Analysis Results

**Text Length:** {len(text)} characters
**Word Count:** {len(text.split())} words
**Status:** Analysis complete
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def connect_gmail(email, password):
    """Mock Gmail connection"""
    if not email or not password:
        return "Please enter email and password", "❌ Not connected"
    
    return f"✅ Connected to {email}", f"✅ Connected"

def scan_emails(query_type, max_emails):
    """Mock email scanning"""
    results = f"""
### Scan Results

**Query Type:** {query_type}
**Max Emails:** {max_emails}
**Status:** Scan complete
**Found:** 5 emails (mock data)

---
✅ Safe - Welcome to Gmail
- From: noreply@gmail.com
- Date: 2024-01-01

⚠️ FRAUD - Urgent: Verify your account
- From: phishing@fake.com
- Date: 2024-01-02

✅ Safe - Monthly Newsletter
- From: newsletter@example.com
- Date: 2024-01-03
"""
    return results

# Create interface
print("Creating Gradio interface...")

with gr.Blocks(title="FraudLens Minimal") as demo:
    gr.Markdown("""
    # 🔍 FraudLens - Minimal Test Version
    Testing Gradio without complex imports
    """)
    
    with gr.Tab("📝 Text Analysis"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter text",
                    placeholder="Type or paste text here...",
                    lines=5
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            
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
                email = gr.Textbox(label="Email")
                password = gr.Textbox(label="Password", type="password")
                connect_btn = gr.Button("Connect")
                status = gr.Textbox(label="Status", interactive=False)
                status_display = gr.Markdown("❌ Not connected")
            
            with gr.Column():
                query = gr.Radio(
                    choices=["Unread", "All", "Recent"],
                    value="Unread",
                    label="Filter"
                )
                max_emails = gr.Slider(1, 100, 10, label="Max Emails")
                scan_btn = gr.Button("Scan")
        
        results = gr.Markdown()
        
        connect_btn.click(
            fn=connect_gmail,
            inputs=[email, password],
            outputs=[status, status_display]
        )
        
        scan_btn.click(
            fn=scan_emails,
            inputs=[query, max_emails],
            outputs=results
        )

print("Interface created successfully!")

if __name__ == "__main__":
    print("Launching app on http://0.0.0.0:7863")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False
    )