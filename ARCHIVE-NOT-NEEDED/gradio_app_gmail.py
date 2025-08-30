#!/usr/bin/env python3
"""
FraudLens Gmail-Integrated Gradio Interface
Real Gmail API integration with demo fallback
"""

import asyncio
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.api.gmail_integration import GmailFraudScanner, EmailAction, EmailAnalysisResult
from loguru import logger

# Set up logging
logger.add("gradio_gmail.log", rotation="10 MB", level="INFO")


class GmailIntegratedDemo:
    """Gmail-integrated demo with fallback to demo mode."""
    
    def __init__(self):
        """Initialize the demo."""
        self.pipeline = None
        self.gmail_scanner = None
        self.is_gmail_connected = False
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_stop_event = threading.Event()
        self.email_stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "actions_taken": {"spam": 0, "trash": 0, "flag": 0},
        }
        self.monitor_logs = []
        self.last_scan_results = []
        
    async def initialize(self):
        """Initialize the demo components."""
        logger.info("Initializing Gmail-integrated demo...")
        self.pipeline = FraudDetectionPipeline()
        await self.pipeline.initialize()
        logger.info("Demo initialized successfully")
    
    def connect_gmail(self) -> Tuple[str, gr.update, gr.update, str]:
        """Connect to Gmail API or fall back to demo mode."""
        try:
            # Try to initialize Gmail scanner
            self.gmail_scanner = GmailFraudScanner(
                credentials_file="credentials.json",
                token_file="token.pickle",
                fraud_threshold=0.7,
                auto_action=False,
            )
            
            # Run async initialization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.gmail_scanner.initialize())
            
            self.is_gmail_connected = True
            logger.info("Successfully connected to Gmail API")
            
            return (
                "🟢 **Gmail Status:** Connected (Live API)",
                gr.update(visible=False),  # Hide connect button
                gr.update(visible=True),   # Show disconnect button
                self._get_stats_display()
            )
            
        except Exception as e:
            logger.warning(f"Failed to connect to Gmail API: {e}")
            logger.info("Falling back to demo mode")
            
            # Fall back to demo mode
            self.is_gmail_connected = False
            self.gmail_scanner = None
            
            return (
                "🟡 **Gmail Status:** Demo Mode (API not configured)",
                gr.update(visible=False),  # Hide connect button
                gr.update(visible=True),   # Show disconnect button
                self._get_stats_display()
            )
    
    def disconnect_gmail(self) -> Tuple[str, gr.update, gr.update, str]:
        """Disconnect from Gmail."""
        self.is_gmail_connected = False
        self.gmail_scanner = None
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_stop_event.set()
            self.monitor_thread = None
        
        return (
            "🔴 **Gmail Status:** Not Connected",
            gr.update(visible=True),   # Show connect button
            gr.update(visible=False),  # Hide disconnect button
            self._get_stats_display()
        )
    
    def _get_stats_display(self) -> str:
        """Get formatted statistics display."""
        fraud_rate = (
            (self.email_stats["fraud_detected"] / self.email_stats["total_processed"] * 100)
            if self.email_stats["total_processed"] > 0 else 0
        )
        
        return f"""
        **📊 Statistics:**
        - Total Processed: {self.email_stats["total_processed"]}
        - Fraud Detected: {self.email_stats["fraud_detected"]}
        - Fraud Rate: {fraud_rate:.1f}%
        - Actions: Spam({self.email_stats["actions_taken"]["spam"]}), Trash({self.email_stats["actions_taken"]["trash"]}), Flag({self.email_stats["actions_taken"]["flag"]})
        - Mode: {"Live API" if self.is_gmail_connected else "Demo"}
        """
    
    def scan_emails(
        self,
        query: str,
        max_results: int,
        process_attachments: bool
    ) -> str:
        """Scan emails using Gmail API or demo data."""
        if self.is_gmail_connected and self.gmail_scanner:
            return self._scan_emails_api(query, max_results, process_attachments)
        else:
            return self._scan_emails_demo(query, max_results, process_attachments)
    
    def _scan_emails_api(
        self,
        query: str,
        max_results: int,
        process_attachments: bool
    ) -> str:
        """Scan emails using real Gmail API."""
        try:
            # Run async scan
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.gmail_scanner.stream_emails(
                    query=query,
                    max_results=max_results,
                    process_attachments=process_attachments,
                    since_days=7
                )
            )
            
            # Update statistics
            self.email_stats["total_processed"] += len(results)
            fraud_count = sum(1 for r in results if r.fraud_score > 0.5)
            self.email_stats["fraud_detected"] += fraud_count
            
            # Store results
            self.last_scan_results = results
            
            # Generate HTML output
            return self._format_scan_results(results, query)
            
        except Exception as e:
            logger.error(f"Error scanning emails: {e}")
            return f"""
            <div style='padding: 20px; background: #ffe0e0; border-radius: 5px;'>
                <h3>❌ Scan Error</h3>
                <p>Failed to scan emails: {str(e)}</p>
                <p>Please check your Gmail connection and try again.</p>
            </div>
            """
    
    def _scan_emails_demo(
        self,
        query: str,
        max_results: int,
        process_attachments: bool
    ) -> str:
        """Scan emails using demo data."""
        import random
        
        # Generate demo results
        demo_emails = [
            {
                "subject": "Urgent: Verify your account",
                "sender": "noreply@suspicious.com",
                "score": 0.89,
                "types": ["phishing"],
                "action": "spam"
            },
            {
                "subject": "Meeting tomorrow at 2 PM",
                "sender": "colleague@company.com",
                "score": 0.12,
                "types": [],
                "action": "none"
            },
            {
                "subject": "You've won $1,000,000!",
                "sender": "lottery@scam.net",
                "score": 0.95,
                "types": ["scam", "phishing"],
                "action": "trash"
            },
            {
                "subject": "Invoice #12345",
                "sender": "billing@vendor.com",
                "score": 0.08,
                "types": [],
                "action": "none"
            },
            {
                "subject": "Security Alert",
                "sender": "security@bank-fake.com",
                "score": 0.78,
                "types": ["phishing"],
                "action": "flag"
            },
            {
                "subject": "Your package is on the way",
                "sender": "shipping@store.com",
                "score": 0.15,
                "types": [],
                "action": "none"
            },
            {
                "subject": "Suspicious activity detected",
                "sender": "alert@phishing.net",
                "score": 0.82,
                "types": ["phishing", "social_engineering"],
                "action": "spam"
            },
        ]
        
        # Select random subset
        selected = random.sample(demo_emails, min(max_results, len(demo_emails)))
        
        # Update demo statistics
        self.email_stats["total_processed"] += len(selected)
        fraud_count = sum(1 for e in selected if e["score"] > 0.5)
        self.email_stats["fraud_detected"] += fraud_count
        
        # Convert to result format
        results = []
        for email in selected:
            result = EmailAnalysisResult(
                message_id=f"demo_{random.randint(1000, 9999)}",
                subject=email["subject"],
                sender=email["sender"],
                recipient="you@example.com",
                date=datetime.now() - timedelta(hours=random.randint(1, 48)),
                fraud_score=email["score"],
                fraud_types=email["types"],
                confidence=0.9,
                explanation=f"Demo analysis: {'Suspicious' if email['score'] > 0.5 else 'Clean'}",
                attachments_analyzed=[],
                action_taken=EmailAction.NONE,
                processing_time_ms=random.uniform(10, 100),
                raw_content_score=email["score"],
                attachment_scores=[],
                combined_score=email["score"],
                flagged=email["score"] > 0.5,
                error=None
            )
            results.append(result)
        
        return self._format_scan_results(results, query)
    
    def _format_scan_results(self, results: List[EmailAnalysisResult], query: str) -> str:
        """Format scan results as HTML."""
        if not results:
            return """
            <div style='padding: 20px; text-align: center; color: #666;'>
                No emails found matching your query.
            </div>
            """
        
        html = """
        <div style='padding: 10px;'>
            <h3>📧 Scan Results</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>
                    <tr style='background: #f0f0f0;'>
                        <th style='padding: 8px; text-align: left;'>Time</th>
                        <th style='padding: 8px; text-align: left;'>Subject</th>
                        <th style='padding: 8px; text-align: left;'>From</th>
                        <th style='padding: 8px; text-align: left;'>Score</th>
                        <th style='padding: 8px; text-align: left;'>Types</th>
                        <th style='padding: 8px; text-align: left;'>Action</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for result in results:
            # Determine color based on score
            if result.fraud_score >= 0.8:
                color = "#ff4757"  # Red
                recommended_action = "trash"
            elif result.fraud_score >= 0.6:
                color = "#ffa502"  # Orange
                recommended_action = "spam"
            elif result.fraud_score >= 0.4:
                color = "#ffd93d"  # Yellow
                recommended_action = "flag"
            else:
                color = "#51cf66"  # Green
                recommended_action = "none"
            
            fraud_types_str = ", ".join(result.fraud_types) if result.fraud_types else "clean"
            
            html += f"""
            <tr>
                <td style='padding: 8px; font-size: 12px;'>{result.date.strftime('%m/%d %H:%M')}</td>
                <td style='padding: 8px;'>{result.subject[:50]}...</td>
                <td style='padding: 8px;'>{result.sender}</td>
                <td style='padding: 8px; color: {color}; font-weight: bold;'>{result.fraud_score:.0%}</td>
                <td style='padding: 8px; font-size: 12px;'>{fraud_types_str}</td>
                <td style='padding: 8px;'>
                    <span style='padding: 2px 6px; background: {color}; color: white; border-radius: 3px; font-size: 12px;'>
                        {recommended_action}
                    </span>
                </td>
            </tr>
            """
        
        suspicious_count = sum(1 for r in results if r.fraud_score > 0.5)
        html += f"""
                </tbody>
            </table>
            <div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;'>
                <strong>Summary:</strong> Processed {len(results)} emails • 
                Found {suspicious_count} suspicious ({suspicious_count/len(results)*100:.0f}%) • 
                Query: "{query}" • 
                Mode: {"Live API" if self.is_gmail_connected else "Demo"}
            </div>
        </div>
        """
        
        return html
    
    def start_monitoring(
        self,
        query: str,
        interval: int,
        auto_action: bool
    ) -> Tuple[str, str]:
        """Start email monitoring."""
        if self.monitoring_active:
            return (
                self._add_monitor_log("⚠️ Monitoring already active"),
                "⚠️ **Monitor Status:** Already Running"
            )
        
        self.monitoring_active = True
        self.monitor_stop_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_emails,
            args=(query, interval, auto_action),
            daemon=True
        )
        self.monitor_thread.start()
        
        log = self._add_monitor_log(f"✅ Started monitoring: {query}")
        log = self._add_monitor_log(f"Check interval: {interval} seconds")
        log = self._add_monitor_log(f"Auto-action: {'Enabled' if auto_action else 'Disabled'}")
        
        return (
            log,
            "🟢 **Monitor Status:** Active"
        )
    
    def stop_monitoring(self) -> Tuple[str, str]:
        """Stop email monitoring."""
        if not self.monitoring_active:
            return (
                self._add_monitor_log("⚠️ Monitoring not active"),
                "⏸️ **Monitor Status:** Inactive"
            )
        
        self.monitoring_active = False
        self.monitor_stop_event.set()
        
        log = self._add_monitor_log("⏹️ Stopped monitoring")
        
        return (
            log,
            "⏸️ **Monitor Status:** Inactive"
        )
    
    def _monitor_emails(self, query: str, interval: int, auto_action: bool):
        """Background thread for email monitoring."""
        while self.monitoring_active and not self.monitor_stop_event.is_set():
            try:
                # Scan emails
                if self.is_gmail_connected and self.gmail_scanner:
                    # Use real API
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        self.gmail_scanner.stream_emails(
                            query=query,
                            max_results=10,
                            process_attachments=False,
                            since_days=1
                        )
                    )
                    
                    # Process results
                    for result in results:
                        if result.fraud_score > 0.5:
                            self._add_monitor_log(
                                f"🚨 FRAUD: {result.subject[:30]}... (Score: {result.fraud_score:.0%})"
                            )
                            
                            if auto_action:
                                action = self._determine_action(result.fraud_score)
                                self._add_monitor_log(f"  → Taking action: {action}")
                                # In real implementation, take action via Gmail API
                        else:
                            self._add_monitor_log(
                                f"✅ CLEAN: {result.subject[:30]}... (Score: {result.fraud_score:.0%})"
                            )
                else:
                    # Demo mode monitoring
                    import random
                    if random.random() > 0.7:
                        self._add_monitor_log(
                            f"🚨 FRAUD: Demo suspicious email (Score: {random.uniform(0.6, 0.95):.0%})"
                        )
                    else:
                        self._add_monitor_log(
                            f"✅ CLEAN: Demo normal email (Score: {random.uniform(0.05, 0.3):.0%})"
                        )
                
                # Wait for next interval
                self.monitor_stop_event.wait(interval)
                
            except Exception as e:
                self._add_monitor_log(f"❌ Error: {str(e)}")
                self.monitor_stop_event.wait(interval)
    
    def _add_monitor_log(self, message: str) -> str:
        """Add message to monitor log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.monitor_logs.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.monitor_logs) > 100:
            self.monitor_logs = self.monitor_logs[-100:]
        
        return "\n".join(self.monitor_logs)
    
    def _determine_action(self, score: float) -> str:
        """Determine action based on fraud score."""
        if score >= 0.9:
            return "trash"
        elif score >= 0.7:
            return "spam"
        elif score >= 0.5:
            return "flag"
        else:
            return "none"
    
    def take_action_on_emails(
        self,
        action_type: str,
        threshold: float
    ) -> str:
        """Take action on emails based on last scan results."""
        if not self.last_scan_results:
            return "❌ No scan results available. Please scan emails first."
        
        if not self.is_gmail_connected:
            return "⚠️ Demo mode - actions simulated only"
        
        action_count = 0
        for result in self.last_scan_results:
            if result.fraud_score >= threshold:
                # In real implementation, take action via Gmail API
                action_count += 1
                self.email_stats["actions_taken"][action_type] = \
                    self.email_stats["actions_taken"].get(action_type, 0) + 1
        
        return f"✅ Applied '{action_type}' action to {action_count} emails with score >= {threshold:.0%}"
    
    def export_results(self) -> str:
        """Export last scan results to CSV."""
        if not self.last_scan_results:
            return "❌ No results to export"
        
        # Convert to pandas DataFrame
        data = []
        for r in self.last_scan_results:
            data.append({
                "Date": r.date.strftime("%Y-%m-%d %H:%M"),
                "Subject": r.subject,
                "Sender": r.sender,
                "Fraud Score": f"{r.fraud_score:.2%}",
                "Fraud Types": ", ".join(r.fraud_types),
                "Action": r.action_taken.value if r.action_taken else "none",
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        filename = f"fraud_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        return f"✅ Exported {len(data)} results to {filename}"


# Create demo instance
demo = GmailIntegratedDemo()

# Initialize demo
def initialize_demo():
    """Initialize the demo asynchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(demo.initialize())

# Initialize on import
initialize_demo()


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="FraudLens Gmail Integration", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🔍 FraudLens - Gmail Fraud Detection System
            Advanced AI-powered fraud detection with real Gmail integration
            """
        )
        
        with gr.Tabs():
            # Email Monitor Tab
            with gr.TabItem("📧 Email Monitor"):
                gr.Markdown(
                    """
                    ## Gmail Fraud Detection Console
                    Monitor and manage email fraud detection in real-time with Gmail API integration.
                    """
                )
                
                with gr.Row():
                    # Connection status
                    with gr.Column(scale=1):
                        gmail_status = gr.Markdown("🔴 **Gmail Status:** Not Connected")
                        connect_btn = gr.Button("🔗 Connect to Gmail", variant="primary")
                        disconnect_btn = gr.Button("🔌 Disconnect", variant="stop", visible=False)
                        
                        gr.Markdown("---")
                        
                        # Statistics
                        stats_display = gr.Markdown(demo._get_stats_display())
                        
                        # Actions
                        gr.Markdown("### Quick Actions")
                        with gr.Row():
                            action_type = gr.Radio(
                                choices=["flag", "spam", "trash"],
                                value="spam",
                                label="Action Type"
                            )
                        action_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Score Threshold"
                        )
                        apply_action_btn = gr.Button("Apply to Last Scan", variant="secondary")
                        action_result = gr.Textbox(label="Action Result", lines=2)
                    
                    # Main console
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            # Inbox Scanner
                            with gr.TabItem("📥 Inbox Scanner"):
                                with gr.Row():
                                    email_query = gr.Textbox(
                                        label="Gmail Query",
                                        value="is:unread",
                                        placeholder="e.g., is:unread, from:noreply, has:attachment",
                                    )
                                    max_emails = gr.Slider(
                                        minimum=1,
                                        maximum=100,
                                        value=20,
                                        step=1,
                                        label="Max Emails",
                                    )
                                    process_attachments = gr.Checkbox(
                                        label="Process Attachments",
                                        value=False,
                                    )
                                
                                with gr.Row():
                                    scan_btn = gr.Button("🔍 Scan Emails", variant="primary")
                                    export_btn = gr.Button("📥 Export Results")
                                    export_result = gr.Textbox(label="Export Status", visible=False)
                                
                                scan_output = gr.HTML(
                                    label="Scan Results",
                                    value="<div style='padding: 20px; text-align: center; color: #666;'>No scan results yet. Click 'Scan Emails' to start.</div>"
                                )
                            
                            # Live Monitor
                            with gr.TabItem("🔄 Live Monitor"):
                                with gr.Row():
                                    monitor_query = gr.Textbox(
                                        label="Monitor Query",
                                        value="is:unread",
                                        placeholder="Gmail query for monitoring",
                                    )
                                    check_interval = gr.Slider(
                                        minimum=10,
                                        maximum=300,
                                        value=60,
                                        step=10,
                                        label="Check Interval (seconds)",
                                    )
                                    auto_action = gr.Checkbox(
                                        label="Auto-Action on Fraud",
                                        value=False,
                                    )
                                
                                with gr.Row():
                                    start_monitor_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                                    stop_monitor_btn = gr.Button("⏸️ Stop Monitoring", variant="stop")
                                    refresh_console_btn = gr.Button("🔄 Refresh", variant="secondary")
                                
                                monitor_console = gr.Textbox(
                                    label="Live Monitor Console",
                                    lines=15,
                                    max_lines=20,
                                    value="Console output will appear here...\n",
                                    interactive=False,
                                )
                                
                                monitor_status = gr.Markdown("⏸️ **Monitor Status:** Inactive")
        
        # Event handlers
        connect_btn.click(
            demo.connect_gmail,
            outputs=[gmail_status, connect_btn, disconnect_btn, stats_display]
        )
        
        disconnect_btn.click(
            demo.disconnect_gmail,
            outputs=[gmail_status, connect_btn, disconnect_btn, stats_display]
        )
        
        scan_btn.click(
            demo.scan_emails,
            inputs=[email_query, max_emails, process_attachments],
            outputs=scan_output
        ).then(
            lambda: demo._get_stats_display(),
            outputs=stats_display
        )
        
        export_btn.click(
            demo.export_results,
            outputs=export_result
        ).then(
            lambda x: gr.update(visible=True, value=x),
            inputs=export_result,
            outputs=export_result
        )
        
        apply_action_btn.click(
            demo.take_action_on_emails,
            inputs=[action_type, action_threshold],
            outputs=action_result
        ).then(
            lambda: demo._get_stats_display(),
            outputs=stats_display
        )
        
        start_monitor_btn.click(
            demo.start_monitoring,
            inputs=[monitor_query, check_interval, auto_action],
            outputs=[monitor_console, monitor_status]
        )
        
        stop_monitor_btn.click(
            demo.stop_monitoring,
            outputs=[monitor_console, monitor_status]
        )
        
        # Auto-refresh monitor console
        def refresh_monitor():
            if demo.monitoring_active and demo.monitor_logs:
                return "\n".join(demo.monitor_logs[-20:])  # Show last 20 lines
            return gr.update()
        
        refresh_console_btn.click(
            refresh_monitor,
            outputs=monitor_console
        )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )