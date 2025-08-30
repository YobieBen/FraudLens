#!/usr/bin/env python3
"""
Simplified FraudLens App for Testing
"""

import gradio as gr
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger.info("Starting simplified FraudLens...")

# Create simple interface
with gr.Blocks(title="FraudLens Simple Test") as demo:
    gr.Markdown("# FraudLens - Simple Test Interface")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Enter any text here...",
                lines=5
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Analysis Result",
                lines=5
            )
    
    def analyze(text):
        if not text:
            return "Please enter some text"
        return f"Analysis complete. Text length: {len(text)} characters"
    
    analyze_btn.click(fn=analyze, inputs=text_input, outputs=output)

logger.info("Interface created, launching...")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False
    )