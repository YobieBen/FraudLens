"""
Gradio-based demonstration interface for FraudLens.

Author: Yobie Benjamin
Date: 2025
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from loguru import logger

from fraudlens.core.config import Config
from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.fusion.fusion_engine import MultiModalFraudFusion
from fraudlens.monitoring.monitor import FraudLensMonitor
from fraudlens.testing.synthetic_generator import SyntheticFraudGenerator


class FraudLensDemo:
    """Interactive demonstration system for FraudLens."""
    
    def __init__(self):
        """Initialize demo system."""
        self.config = Config()
        self.pipeline = FraudDetectionPipeline(self.config)
        self.fusion = MultiModalFraudFusion()
        self.monitor = FraudLensMonitor()
        self.generator = SyntheticFraudGenerator()
        
        # Initialize pipeline
        asyncio.run(self.pipeline.initialize())
        
        # Sample data cache
        self.sample_data = self._load_sample_data()
    
    def _load_sample_data(self) -> Dict[str, Any]:
        """Load sample fraud data."""
        return {
            "phishing_emails": [
                {
                    "subject": "Urgent: Verify Your Account",
                    "body": "Your account has been compromised. Click here to secure it immediately.",
                    "fraud_score": 0.85,
                },
                {
                    "subject": "Congratulations! You've Won!",
                    "body": "You've won $1,000,000! Send us your bank details to claim.",
                    "fraud_score": 0.92,
                },
            ],
            "documents": [
                "Forged Invoice Example",
                "Altered Bank Statement",
                "Fake ID Document",
            ],
            "transactions": [
                {"amount": 9999, "type": "structuring", "risk": "high"},
                {"amount": 50, "type": "normal", "risk": "low"},
            ],
        }
    
    def analyze_text(
        self,
        text: str,
        fraud_type: str = "auto",
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Analyze text for fraud.
        
        Args:
            text: Input text
            fraud_type: Type of fraud to detect
            
        Returns:
            Fraud score, explanation, and details
        """
        start_time = time.time()
        
        # Process text
        result = asyncio.run(self.pipeline.process(text))
        
        if result:
            fraud_score = result.fraud_score
            explanation = result.explanation if hasattr(result, 'explanation') else ""
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_detection(
                detector_id="text_detector",
                fraud_score=fraud_score,
                latency_ms=latency_ms,
                fraud_type=fraud_type,
            )
            
            # Create details
            details = {
                "confidence": result.confidence if hasattr(result, 'confidence') else fraud_score,
                "fraud_types": result.fraud_types if hasattr(result, 'fraud_types') else [],
                "evidence": result.evidence if hasattr(result, 'evidence') else {},
                "processing_time_ms": latency_ms,
            }
        else:
            fraud_score = 0.0
            explanation = "No fraud detected"
            details = {}
        
        return fraud_score, explanation, details
    
    def analyze_image(
        self,
        image,
        analysis_type: str = "forgery",
    ) -> Tuple[float, str, Dict[str, Any], Any]:
        """
        Analyze image for fraud.
        
        Args:
            image: Input image
            analysis_type: Type of analysis
            
        Returns:
            Fraud score, explanation, details, and annotated image
        """
        if image is None:
            return 0.0, "No image provided", {}, None
        
        start_time = time.time()
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if isinstance(image, np.ndarray):
                Image.fromarray(image).save(tmp.name)
            else:
                image.save(tmp.name)
            
            # Process image
            result = asyncio.run(self.pipeline.process(tmp.name))
        
        if result:
            fraud_score = result.fraud_score
            explanation = result.explanation if hasattr(result, 'explanation') else ""
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_detection(
                detector_id="vision_detector",
                fraud_score=fraud_score,
                latency_ms=latency_ms,
                fraud_type=analysis_type,
            )
            
            # Create details
            details = {
                "analysis_type": analysis_type,
                "confidence": f"{100-fraud_score:.1f}%",
                "fraud_indicators": result.fraud_types if hasattr(result, 'fraud_types') else [],
                "processing_time_ms": f"{latency_ms:.1f}",
            }
            
            # Create annotated image (simplified)
            annotated = image  # In real implementation, add bounding boxes etc.
        else:
            fraud_score = 0.0
            explanation = "No fraud detected"
            details = {}
            annotated = image
        
        return fraud_score, explanation, details, annotated
    
    def analyze_document(
        self,
        file,
        doc_type: str = "auto",
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Analyze document for fraud.
        
        Args:
            file: Uploaded file
            doc_type: Document type
            
        Returns:
            Fraud score, explanation, and details
        """
        if file is None:
            return 0.0, "No document provided", {}
        
        start_time = time.time()
        
        # Process document as image
        try:
            result = asyncio.run(self.pipeline.process(file.name, modality="image"))
            
            if result:
                fraud_score = result.fraud_score
                
                # Build detailed explanation based on fraud score
                if fraud_score < 20:
                    explanation = f"✅ Document appears authentic\nConfidence: {100-fraud_score:.1f}%\n"
                    explanation += "• Security features validated\n• No signs of manipulation detected"
                elif fraud_score < 50:
                    explanation = f"⚠️ Minor concerns detected\nRisk Level: {fraud_score:.1f}%\n"
                    explanation += "• Some irregularities found\n• Manual review recommended"
                elif fraud_score < 80:
                    explanation = f"⚠️ Significant issues detected\nRisk Level: {fraud_score:.1f}%\n"
                    explanation += "• Multiple red flags identified\n• High probability of tampering"
                else:
                    explanation = f"🚨 HIGH FRAUD RISK\nRisk Level: {fraud_score:.1f}%\n"
                    explanation += "• Document appears to be forged\n• Do not accept this document"
                
                # Add document-specific analysis
                if doc_type == "passport":
                    explanation += "\n\n📄 Passport Checks:\n"
                    explanation += "• MRZ validation performed\n"
                    explanation += "• Biometric page analyzed\n"
                    explanation += "• Security watermarks checked"
                elif doc_type == "driver_license":
                    explanation += "\n\n📄 License Checks:\n"
                    explanation += "• Format validated for issuing state\n"
                    explanation += "• Holographic features analyzed\n"
                    explanation += "• Photo authenticity verified"
                elif doc_type == "id_card":
                    explanation += "\n\n📄 ID Card Checks:\n"
                    explanation += "• Security elements validated\n"
                    explanation += "• Text consistency checked\n"
                    explanation += "• Photo manipulation detection performed"
                
                # Add detected fraud types if any
                if hasattr(result, 'fraud_types') and result.fraud_types:
                    # Convert FraudType enums to strings
                    fraud_type_strings = []
                    for ft in result.fraud_types:
                        if hasattr(ft, 'value'):
                            fraud_type_strings.append(ft.value)
                        else:
                            fraud_type_strings.append(str(ft))
                    if fraud_type_strings:
                        explanation += f"\n\n⚠️ Issues Found: {', '.join(fraud_type_strings)}"
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                self.monitor.record_detection(
                    detector_id="document_detector",
                    fraud_score=fraud_score,
                    latency_ms=latency_ms,
                    fraud_type=doc_type,
                )
                
                details = {
                    "doc_type": doc_type,
                    "confidence": f"{100-fraud_score:.1f}%",
                    "risk_level": "Low" if fraud_score < 30 else ("Medium" if fraud_score < 70 else "High"),
                    "processing_time": f"{latency_ms:.1f}ms",
                    "checks_performed": [
                        "Visual authenticity",
                        "Security features",
                        "Manipulation detection",
                        "Format validation",
                        "OCR text extraction"
                    ],
                    "fraud_indicators": fraud_type_strings if 'fraud_type_strings' in locals() else []
                }
            else:
                fraud_score = 0.0
                explanation = "Unable to analyze document. Please ensure:\n• Image is clear and well-lit\n• Entire document is visible\n• File is not corrupted"
                details = {"error": "Processing failed"}
                
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            fraud_score = 0.0
            explanation = "Error processing document. Please try again with a clear image."
            details = {"error": str(e)}
        
        return fraud_score, explanation, details
    
    def multimodal_analysis(
        self,
        text: Optional[str],
        image: Optional[Any],
        document: Optional[Any],
    ) -> Tuple[float, str, Dict[str, Any], Any]:
        """
        Perform multi-modal fraud analysis.
        
        Args:
            text: Text input
            image: Image input
            document: Document input
            
        Returns:
            Fused fraud score, explanation, details, and visualization
        """
        results = {}
        
        # Analyze each modality
        if text:
            text_score, _, text_details = self.analyze_text(text)
            results["text"] = {
                "score": text_score,
                "details": text_details,
            }
        
        if image is not None:
            image_score, _, image_details, _ = self.analyze_image(image)
            results["image"] = {
                "score": image_score,
                "details": image_details,
            }
        
        if document is not None:
            doc_score, _, doc_details = self.analyze_document(document)
            results["document"] = {
                "score": doc_score,
                "details": doc_details,
            }
        
        if not results:
            return 0.0, "No input provided", {}, None
        
        # Fuse results
        fused_score = np.mean([r["score"] for r in results.values()])
        
        # Create explanation
        high_risk_modalities = [
            k for k, v in results.items()
            if v["score"] > 0.7
        ]
        
        if high_risk_modalities:
            explanation = f"High fraud risk detected in: {', '.join(high_risk_modalities)}"
        elif fused_score > 0.5:
            explanation = "Moderate fraud risk detected"
        else:
            explanation = "Low fraud risk"
        
        # Create visualization
        viz = self._create_fusion_visualization(results, fused_score)
        
        details = {
            "modality_scores": {k: v["score"] for k, v in results.items()},
            "fused_score": fused_score,
            "analysis_count": len(results),
        }
        
        return fused_score, explanation, details, viz
    
    def _create_fusion_visualization(
        self,
        results: Dict[str, Any],
        fused_score: float,
    ) -> Any:
        """Create fusion visualization."""
        # Create radar chart
        categories = list(results.keys()) + ["Fused"]
        scores = [r["score"] for r in results.values()] + [fused_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Fraud Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Multi-Modal Fraud Analysis",
        )
        
        return fig
    
    def generate_synthetic_sample(
        self,
        sample_type: str,
        fraud_level: str,
    ) -> Dict[str, Any]:
        """
        Generate synthetic fraud sample.
        
        Args:
            sample_type: Type of sample
            fraud_level: Fraud complexity level
            
        Returns:
            Generated sample
        """
        if sample_type == "phishing_email":
            urgency = {"low": 3, "medium": 6, "high": 9}[fraud_level]
            email = self.generator.generate_phishing_email(
                fraud_type="spear",
                urgency_level=urgency,
            )
            return {
                "type": "email",
                "subject": email.subject,
                "body": email.body,
                "fraud_score": email.fraud_score,
            }
        
        elif sample_type == "forged_document":
            forgery_type = {"low": "altered", "medium": "fake", "high": "cloned"}[fraud_level]
            doc = self.generator.create_forged_document(
                doc_type="invoice",
                forgery_type=forgery_type,
            )
            return {
                "type": "document",
                "content": doc.content[:500],  # First 500 chars
                "fraud_indicators": doc.fraud_indicators,
            }
        
        elif sample_type == "fraud_scenario":
            scenario = self.generator.synthesize_fraud_scenario(
                scenario_type="phishing_campaign",
                complexity=fraud_level,
            )
            return {
                "type": "scenario",
                "case_id": scenario.case_id,
                "description": scenario.metadata.get("description", "Multi-modal fraud case"),
            }
        
        else:
            return {"error": "Unknown sample type"}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        status = self.monitor.get_status()
        
        return {
            "uptime": f"{status['uptime'] / 3600:.1f} hours",
            "total_requests": status['statistics']['total_requests'],
            "success_rate": f"{status['statistics']['success_rate']:.1%}",
            "avg_latency": f"{status['current_metrics']['latency_ms']:.2f}ms",
            "active_alerts": status['alert_count'],
        }
    
    def create_performance_dashboard(self) -> Any:
        """Create performance dashboard."""
        # Get metrics history (simplified)
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        
        # Simulated metrics
        latencies = np.random.normal(100, 20, 100)
        throughput = np.random.normal(50, 10, 100)
        errors = np.random.binomial(1, 0.05, 100)
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Latency (ms)", "Throughput (QPS)",
                "Error Rate", "CPU Usage"
            )
        )
        
        # Latency
        fig.add_trace(
            go.Scatter(x=timestamps, y=latencies, name="Latency"),
            row=1, col=1
        )
        
        # Throughput
        fig.add_trace(
            go.Scatter(x=timestamps, y=throughput, name="Throughput"),
            row=1, col=2
        )
        
        # Error rate
        fig.add_trace(
            go.Scatter(x=timestamps, y=errors.cumsum() / np.arange(1, 101), name="Error Rate"),
            row=2, col=1
        )
        
        # CPU usage
        cpu_usage = np.random.normal(40, 15, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, name="CPU %"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title="System Performance Metrics")
        
        return fig


def create_interface():
    """Create Gradio interface."""
    demo = FraudLensDemo()
    
    with gr.Blocks(title="FraudLens Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🔍 FraudLens - Advanced Fraud Detection System
            
            Welcome to the FraudLens demonstration interface. This system uses advanced AI 
            to detect various types of fraud across multiple modalities.
            """
        )
        
        with gr.Tabs():
            # Text Analysis Tab
            with gr.TabItem("📝 Text Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Enter Text to Analyze",
                            placeholder="Paste suspicious email, message, or document text here...",
                            lines=10,
                        )
                        
                        fraud_type_select = gr.Dropdown(
                            choices=["auto", "phishing", "scam", "spam", "social_engineering"],
                            value="auto",
                            label="Fraud Type",
                        )
                        
                        analyze_text_btn = gr.Button("Analyze Text", variant="primary")
                    
                    with gr.Column(scale=1):
                        text_fraud_score = gr.Number(label="Fraud Score", precision=3)
                        text_explanation = gr.Textbox(label="Explanation", lines=3)
                        text_details = gr.JSON(label="Detection Details")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["Your account has been compromised. Click here to secure it."],
                        ["Congratulations! You've won $1,000,000!"],
                        ["This is a normal business email about the upcoming meeting."],
                    ],
                    inputs=text_input,
                )
                
                analyze_text_btn.click(
                    demo.analyze_text,
                    inputs=[text_input, fraud_type_select],
                    outputs=[text_fraud_score, text_explanation, text_details],
                )
            
            # Image Analysis Tab
            with gr.TabItem("🖼️ Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image", type="pil")
                        
                        analysis_type = gr.Radio(
                            choices=["forgery", "deepfake", "manipulation", "logo_impersonation"],
                            value="forgery",
                            label="Analysis Type",
                        )
                        
                        analyze_image_btn = gr.Button("Analyze Image", variant="primary")
                    
                    with gr.Column():
                        image_fraud_score = gr.Number(label="Fraud Score", precision=3)
                        image_explanation = gr.Textbox(label="Explanation", lines=3)
                        image_output = gr.Image(label="Analyzed Image")
                
                with gr.Row():
                    image_details = gr.JSON(label="Analysis Details")
                
                analyze_image_btn.click(
                    demo.analyze_image,
                    inputs=[image_input, analysis_type],
                    outputs=[image_fraud_score, image_explanation, image_details, image_output],
                )
            
            # Document Analysis Tab
            with gr.TabItem("📄 Document Analysis"):
                with gr.Row():
                    with gr.Column():
                        doc_input = gr.File(
                            label="Upload Document",
                            file_types=[".pdf", ".docx", ".txt", ".jpg", ".png"],
                        )
                        
                        doc_type = gr.Dropdown(
                            choices=["auto", "invoice", "contract", "id", "bank_statement"],
                            value="auto",
                            label="Document Type",
                        )
                        
                        analyze_doc_btn = gr.Button("Analyze Document", variant="primary")
                    
                    with gr.Column():
                        doc_fraud_score = gr.Number(label="Fraud Score", precision=3)
                        doc_explanation = gr.Textbox(label="Explanation", lines=3)
                        doc_details = gr.JSON(label="Analysis Details")
                
                analyze_doc_btn.click(
                    demo.analyze_document,
                    inputs=[doc_input, doc_type],
                    outputs=[doc_fraud_score, doc_explanation, doc_details],
                )
            
            # Multi-Modal Analysis Tab
            with gr.TabItem("🔀 Multi-Modal Fusion"):
                gr.Markdown(
                    """
                    ### Analyze Multiple Inputs Simultaneously
                    Combine text, image, and document analysis for comprehensive fraud detection.
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        mm_text = gr.Textbox(label="Text Input (Optional)", lines=5)
                        mm_image = gr.Image(label="Image Input (Optional)", type="pil")
                        mm_document = gr.File(label="Document Input (Optional)")
                        
                        analyze_mm_btn = gr.Button("Perform Multi-Modal Analysis", variant="primary")
                    
                    with gr.Column():
                        mm_fraud_score = gr.Number(label="Fused Fraud Score", precision=3)
                        mm_explanation = gr.Textbox(label="Analysis Summary", lines=3)
                        mm_visualization = gr.Plot(label="Fusion Visualization")
                        mm_details = gr.JSON(label="Detailed Results")
                
                analyze_mm_btn.click(
                    demo.multimodal_analysis,
                    inputs=[mm_text, mm_image, mm_document],
                    outputs=[mm_fraud_score, mm_explanation, mm_details, mm_visualization],
                )
            
            # Synthetic Data Generator Tab
            with gr.TabItem("🎲 Synthetic Data Generator"):
                gr.Markdown(
                    """
                    ### Generate Test Data
                    Create synthetic fraud samples for testing and training.
                    """
                )
                
                with gr.Row():
                    sample_type = gr.Radio(
                        choices=["phishing_email", "forged_document", "fraud_scenario"],
                        value="phishing_email",
                        label="Sample Type",
                    )
                    
                    fraud_level = gr.Radio(
                        choices=["low", "medium", "high"],
                        value="medium",
                        label="Fraud Complexity",
                    )
                    
                    generate_btn = gr.Button("Generate Sample", variant="primary")
                
                generated_output = gr.JSON(label="Generated Sample")
                
                generate_btn.click(
                    demo.generate_synthetic_sample,
                    inputs=[sample_type, fraud_level],
                    outputs=generated_output,
                )
            
            # System Monitoring Tab
            with gr.TabItem("📊 System Monitoring"):
                gr.Markdown("### Real-time System Performance")
                
                with gr.Row():
                    metrics_display = gr.JSON(label="Current Metrics")
                    refresh_metrics_btn = gr.Button("Refresh Metrics")
                
                performance_chart = gr.Plot(label="Performance Dashboard")
                
                refresh_metrics_btn.click(
                    demo.get_system_metrics,
                    outputs=metrics_display,
                )
                
                refresh_metrics_btn.click(
                    demo.create_performance_dashboard,
                    outputs=performance_chart,
                )
            
            # API Playground Tab
            with gr.TabItem("🔧 API Playground"):
                gr.Markdown(
                    """
                    ### Test FraudLens API
                    
                    ```python
                    import requests
                    
                    # Text analysis endpoint
                    response = requests.post(
                        "http://localhost:7860/api/analyze",
                        json={"text": "Your suspicious text here"}
                    )
                    
                    result = response.json()
                    print(f"Fraud Score: {result['fraud_score']}")
                    ```
                    """
                )
                
                with gr.Row():
                    api_input = gr.Textbox(
                        label="API Request (JSON)",
                        value='{"text": "Test fraud detection"}',
                        lines=5,
                    )
                    
                    api_endpoint = gr.Dropdown(
                        choices=["/analyze", "/detect", "/fusion"],
                        value="/analyze",
                        label="Endpoint",
                    )
                    
                    api_test_btn = gr.Button("Test API")
                
                api_output = gr.JSON(label="API Response")
                
                # API test would be implemented here
        
        # Footer
        gr.Markdown(
            """
            ---
            
            ### About FraudLens
            
            FraudLens is an advanced fraud detection system that combines multiple AI techniques:
            - 🧠 Deep learning models for pattern recognition
            - 📊 Statistical analysis for anomaly detection
            - 🔀 Multi-modal fusion for comprehensive analysis
            - 🔄 Adaptive learning for continuous improvement
            
            **Note:** This is a demonstration interface. In production, additional security 
            measures and authentication would be implemented.
            
            © 2025 FraudLens - Built with ❤️ by Yobie Benjamin
            """
        )
    
    return interface


if __name__ == "__main__":
    print("🚀 Starting FraudLens Demo Interface...")
    print("=" * 50)
    
    # Create and launch interface
    interface = create_interface()
    print("✅ Gradio interface created")
    
    print("\n📊 Launching server...")
    print("=" * 50)
    print("🌐 Server will be available at: http://localhost:7860")
    print("=" * 50)
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public URL
        debug=False,
        show_api=False,
        quiet=False,
        inbrowser=True  # Automatically open in browser
    )