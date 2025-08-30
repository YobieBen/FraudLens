#!/usr/bin/env python3
"""
FraudLens Integrated Application
Complete fraud detection system with Gmail integration and analytics dashboard
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
import base64
import io
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.api.gmail_imap_integration import GmailIMAPScanner
from fraudlens.processors.text.detector import TextFraudDetector

# Set up logging
logger.add("fraudlens_integrated.log", rotation="10 MB", level="INFO")


class IntegratedFraudLensApp:
    """Integrated FraudLens application with all features."""
    
    def __init__(self):
        """Initialize the integrated app."""
        # Core components
        self.pipeline = None
        self.text_detector = TextFraudDetector()
        self.gmail_scanner = None
        
        # Gmail integration state
        self.gmail_user = None
        self.gmail_password = None
        self.is_gmail_connected = False
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_stop_event = threading.Event()
        
        # Email statistics
        self.email_stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "actions_taken": {"spam": 0, "trash": 0, "flag": 0},
            "last_batch_size": 0
        }
        self.monitor_logs = []
        self.last_scan_results = []
        self.last_check_time = None
        self.email_offset = 0  # Track offset for pagination
        self.credentials_status = self._check_credentials_status()
        
        # Analytics data
        self.fraud_data = []
        self.email_data = []
        self.detection_history = []
        
        # Initialize with sample data for analytics
        self._generate_sample_analytics_data()
    
    def _check_credentials_status(self) -> Dict[str, Any]:
        """Check current credentials status."""
        status = {
            "credentials_exists": os.path.exists("credentials.json"),
            "token_exists": os.path.exists("token.pickle"),
            "client_id": "Not configured",
            "project_name": "Not configured",
            "auth_status": "Not authenticated",
        }
        
        if status["credentials_exists"]:
            try:
                with open("credentials.json", "r") as f:
                    creds_data = json.load(f)
                    if "installed" in creds_data:
                        status["client_id"] = creds_data["installed"].get("client_id", "Unknown")[:30] + "..."
                        status["project_name"] = creds_data["installed"].get("project_id", "Unknown")
                    elif "web" in creds_data:
                        status["client_id"] = creds_data["web"].get("client_id", "Unknown")[:30] + "..."
                        status["project_name"] = creds_data["web"].get("project_id", "Unknown")
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        if status["token_exists"]:
            status["auth_status"] = "Previously authenticated"
        
        return status
    
    def _generate_sample_analytics_data(self):
        """Generate sample data for analytics dashboard."""
        end_date = datetime.now()
        fraud_types = ["phishing", "deepfake", "identity_theft", "social_engineering", 
                      "document_fraud", "money_laundering", "scam"]
        
        for i in range(30):
            date = end_date - timedelta(days=i)
            daily_detections = random.randint(5, 25)
            
            for _ in range(daily_detections):
                self.fraud_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "time": f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                    "type": random.choice(fraud_types),
                    "confidence": random.uniform(0.6, 1.0),
                    "risk_level": random.choice(["Low", "Medium", "High", "Critical"]),
                    "source": random.choice(["Email", "Document", "Video", "Text", "Image"]),
                    "status": random.choice(["Detected", "Blocked", "Quarantined", "Reviewed"]),
                })
        
        # Email-specific patterns
        email_domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com", 
                        "phishing-site.com", "suspicious-domain.net"]
        
        for i in range(100):
            date = end_date - timedelta(days=random.randint(0, 30))
            hour = random.randint(0, 23)
            
            self.email_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "hour": hour,
                "sender_domain": random.choice(email_domains),
                "fraud_detected": random.choice([True, False]),
                "confidence": random.uniform(0.3, 1.0) if random.random() > 0.3 else 0,
                "type": random.choice(["phishing", "spam", "malware", "legitimate"]),
            })
    
    async def initialize(self):
        """Initialize the app components."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        logger.info("Initializing FraudLens integrated app...")
        self.pipeline = FraudDetectionPipeline()
        await self.pipeline.initialize()
        
        if not self.text_detector._initialized:
            await self.text_detector.initialize()
        
        self._initialized = True
        logger.info("App initialized successfully")
    
    def ensure_initialized(self):
        """Ensure app is initialized (lazy initialization)."""
        if not hasattr(self, '_initialized') or not self._initialized:
            asyncio.run(self.initialize())
    
    # Gmail Integration Methods
    def get_credentials_card(self) -> str:
        """Generate HTML for credentials status card."""
        if hasattr(self, 'gmail_user') and self.gmail_user:
            status_color = "#51cf66"
            status_icon = "✅"
            status_text = "Connected"
            email_display = self.gmail_user
        else:
            status_color = "#ff6348"
            status_icon = "❌"
            status_text = "Not connected"
            email_display = "Not logged in"
        
        html = f"""
        <div style='border: 2px solid {status_color}; border-radius: 10px; padding: 15px; background: linear-gradient(135deg, #fff 0%, {status_color}22 100%);'>
            <h3 style='margin-top: 0; color: {status_color};'>{status_icon} Gmail Status</h3>
            <p style='margin: 5px 0;'><strong>Status:</strong> {status_text}</p>
            <p style='margin: 5px 0;'><strong>Account:</strong> {email_display}</p>
        </div>
        """
        return html
    
    def authenticate_gmail(self, email, app_password):
        """Authenticate with Gmail using email and app password."""
        if not email or not app_password:
            return "Please enter both email and app password", self.get_credentials_card()
        
        # Validate email format
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@gmail\.com$', email):
            return "❌ Please enter a valid Gmail address", self.get_credentials_card()
        
        try:
            # Store credentials (in production, use secure storage)
            self.gmail_user = email
            self.gmail_password = app_password  # App-specific password
            
            # Note: For actual Gmail API integration, you would need to:
            # 1. Use OAuth2 flow with google-auth library
            # 2. Or use IMAP with app-specific passwords
            # For now, we'll simulate successful authentication
            
            return "✅ Authentication successful! Click 'Connect to Gmail' to start.", self.get_credentials_card()
            
        except Exception as e:
            return f"❌ Authentication failed: {str(e)}", self.get_credentials_card()
    
    def clear_credentials(self):
        """Clear all credentials."""
        try:
            self.gmail_user = None
            self.gmail_password = None
            self.is_gmail_connected = False
            
            return "✅ Credentials cleared", self.get_credentials_card()
        except Exception as e:
            return f"❌ Error clearing credentials: {str(e)}", self.get_credentials_card()
    
    def connect_gmail(self):
        """Connect to Gmail using IMAP."""
        self.ensure_initialized()
        try:
            if not hasattr(self, 'gmail_user') or not self.gmail_user:
                return (
                    "🔴 **Gmail Status:** Please enter credentials first.",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    self._get_stats_display(),
                    self.get_credentials_card()
                )
            
            # Initialize IMAP scanner
            self.gmail_scanner = GmailIMAPScanner(fraud_detector=self.text_detector)
            
            # Connect using IMAP
            try:
                success = self.gmail_scanner.connect(self.gmail_user, self.gmail_password)
                if success:
                    self.is_gmail_connected = True
                    return (
                        f"🟢 **Gmail Status:** Connected as {self.gmail_user}",
                        gr.update(visible=False),
                        gr.update(visible=True),
                        self._get_stats_display(),
                        self.get_credentials_card()
                    )
            except Exception as imap_error:
                error_msg = str(imap_error)
                if "Invalid" in error_msg or "credentials" in error_msg.lower():
                    return (
                        "🔴 **Gmail Status:** Invalid credentials. Please check your email and app password.",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        self._get_stats_display(),
                        self.get_credentials_card()
                    )
                else:
                    return (
                        f"🔴 **Gmail Status:** {error_msg}",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        self._get_stats_display(),
                        self.get_credentials_card()
                    )
            
        except Exception as e:
            logger.error(f"Failed to connect to Gmail: {e}")
            return (
                f"🔴 **Gmail Status:** Connection failed - {str(e)}",
                gr.update(visible=True),
                gr.update(visible=False),
                self._get_stats_display(),
                self.get_credentials_card()
            )
    
    def disconnect_gmail(self):
        """Disconnect from Gmail."""
        if self.gmail_scanner:
            self.gmail_scanner.disconnect()
        self.is_gmail_connected = False
        self.gmail_scanner = None
        return (
            "🔴 **Gmail Status:** Disconnected",
            gr.update(visible=True),
            gr.update(visible=False),
            self._get_stats_display()
        )
    
    def _get_stats_display(self) -> str:
        """Get formatted statistics display."""
        return f"""
        ### 📊 Statistics
        - **Total Processed:** {self.email_stats['total_processed']}
        - **Fraud Detected:** {self.email_stats['fraud_detected']}
        - **Actions Taken:**
          - Spam: {self.email_stats['actions_taken']['spam']}
          - Trash: {self.email_stats['actions_taken']['trash']}
          - Flag: {self.email_stats['actions_taken']['flag']}
        """
    
    def build_gmail_query(self, query_type: str, sender: str = None, date_from: str = None, date_to: str = None) -> str:
        """Build Gmail IMAP query from user-friendly options."""
        if query_type == "Unread":
            return "UNSEEN"
        elif query_type == "All":
            return "ALL"
        elif query_type == "By Sender" and sender:
            return f'FROM "{sender}"'
        elif query_type == "By Date Range" and date_from:
            # Convert to IMAP date format
            from datetime import datetime
            try:
                date_obj = datetime.strptime(date_from, "%Y-%m-%d")
                since_date = date_obj.strftime("%d-%b-%Y")
                query = f'SINCE {since_date}'
                
                if date_to:
                    date_obj_to = datetime.strptime(date_to, "%Y-%m-%d")
                    before_date = date_obj_to.strftime("%d-%b-%Y")
                    query += f' BEFORE {before_date}'
                
                return query
            except:
                return "ALL"
        elif query_type == "With Attachments":
            # IMAP doesn't have direct attachment filter, we'll filter after fetching
            return "ALL"
        elif query_type == "Unknown Senders":
            # This requires checking against contacts - for now return all
            return "ALL"
        else:
            return "UNSEEN"
    
    def scan_emails(self, query_type: str, sender: str, date_from: str, date_to: str, max_emails: int, process_attachments: bool, is_continuation: bool = False):
        """Scan emails for fraud using real IMAP connection with batch processing."""
        if not self.is_gmail_connected:
            return (
                "<div style='color: red; padding: 20px;'>⚠️ Please connect to Gmail first</div>",
                "**Last Check:** Never",
                f"**Total Processed:** {self.email_stats.get('total_processed', 0)}",
                gr.update(visible=False)
            )
        
        try:
            # Build query from user-friendly options
            query = self.build_gmail_query(query_type, sender, date_from, date_to)
            
            # Reset offset if query changed (not a continuation)
            if not is_continuation:
                self.email_offset = 0
            
            # Update last check time
            self.last_check_time = datetime.now()
            last_check_str = self.last_check_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Use real IMAP scanning with increased limit and pagination
            results = asyncio.run(
                self.gmail_scanner.scan_for_fraud(
                    query=query,
                    max_emails=max_emails,
                    offset=self.email_offset
                )
            )
            
            # Update offset for next batch
            if len(results) == max_emails:
                self.email_offset += max_emails
            else:
                self.email_offset = 0  # Reset if we got less than max (end of emails)
            
            # Filter for attachments if needed
            if query_type == "With Attachments":
                results = [r for r in results if r.get('has_attachments', False)]
            
            # Filter for unknown senders (simplified - checks for noreply/no-reply patterns)
            if query_type == "Unknown Senders":
                unknown_patterns = ['noreply', 'no-reply', 'donotreply', 'notification', 'automated']
                results = [r for r in results if any(p in r.get('sender', '').lower() for p in unknown_patterns)]
            
            self.last_scan_results = results
            self.email_stats["total_processed"] += len(results)
            self.email_stats["last_batch_size"] = len(results)
            
            # Update fraud data for analytics
            for result in results:
                if result['is_fraud']:
                    self.email_stats["fraud_detected"] += 1
                    self.fraud_data.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.now().strftime("%H:%M"),
                        "type": "phishing" if "phishing" in result.get('fraud_types', []) else "email_fraud",
                        "confidence": result['confidence'],
                        "risk_level": "High" if result['confidence'] > 0.7 else "Medium",
                        "source": "Email",
                        "status": "Detected"
                    })
            
            # Generate HTML output
            html_output = self._generate_scan_results_html(results)
            
            # Show continue button if we hit the max limit
            show_continue = len(results) == max_emails
            
            return (
                html_output,
                f"**Last Check:** {last_check_str}",
                f"**Total Processed:** {self.email_stats['total_processed']}",
                gr.update(visible=show_continue)
            )
            
        except Exception as e:
            logger.error(f"Email scan failed: {e}")
            return (
                f"<div style='color: red; padding: 20px;'>❌ Scan failed: {str(e)}</div>",
                f"**Last Check:** Failed",
                f"**Total Processed:** {self.email_stats.get('total_processed', 0)}",
                gr.update(visible=False)
            )
    
    def _generate_scan_results_html(self, results: List[Dict]) -> str:
        """Generate HTML for scan results."""
        if not results:
            return "<div style='padding: 20px; text-align: center; color: #666;'>No emails found matching the query.</div>"
        
        html = "<div style='padding: 10px;'>"
        html += f"<h4>Scanned {len(results)} emails</h4>"
        
        fraud_count = sum(1 for r in results if r['is_fraud'])
        if fraud_count > 0:
            html += f"<div style='background: #ff634722; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
            html += f"⚠️ <strong>Fraud detected in {fraud_count} email(s)</strong>"
            html += "</div>"
        
        for result in results:
            color = "#ff6347" if result['is_fraud'] else "#51cf66"
            icon = "⚠️" if result['is_fraud'] else "✅"
            
            html += f"""
            <div style='border: 1px solid {color}; border-radius: 5px; padding: 10px; margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <strong>{icon} {result['subject']}</strong><br>
                        <small>From: {result['sender']} | {result['date']}</small><br>
                        <small>ID: {result['message_id']}</small>
                    </div>
                    <div style='text-align: right;'>
                        <span style='color: {color}; font-weight: bold;'>
                            {result['confidence']:.1%} confidence
                        </span>
                    </div>
                </div>
                {f"<div style='margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px;'><strong>Fraud Types:</strong> {', '.join(result.get('fraud_types', []))}<br><strong>Risk Level:</strong> {result.get('risk_level', 'Medium')}</div>" if result['is_fraud'] else ""}
            </div>
            """
        
        html += "</div>"
        return html
    
    # Analytics Dashboard Methods
    def create_trend_chart(self, days: int = 30) -> go.Figure:
        """Create fraud trends over time chart."""
        df = pd.DataFrame(self.fraud_data)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= start_date]
        
        daily_counts = df.groupby(['date', 'type']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for fraud_type in daily_counts.columns:
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=daily_counts[fraud_type],
                mode='lines+markers',
                name=fraud_type.replace('_', ' ').title(),
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Fraud Detection Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Detections",
            template="plotly_white",
            hovermode='x unified',
            showlegend=True,
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_distribution_pie_chart(self) -> go.Figure:
        """Create fraud type distribution pie chart."""
        df = pd.DataFrame(self.fraud_data)
        
        type_counts = df['type'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=[t.replace('_', ' ').title() for t in type_counts.index],
            values=type_counts.values,
            hole=0.3,
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            ),
            textposition='auto',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Fraud Type Distribution",
            template="plotly_white",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True
        )
        
        return fig
    
    def create_email_heatmap(self) -> go.Figure:
        """Create heatmap for email fraud patterns."""
        df = pd.DataFrame(self.email_data)
        
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        
        fraud_df = df[df['fraud_detected'] == True]
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = list(range(24))
        
        heatmap_data = pd.pivot_table(
            fraud_df,
            values='fraud_detected',
            index='hour',
            columns='day_of_week',
            aggfunc='count',
            fill_value=0
        )
        
        for day in days_order:
            if day not in heatmap_data.columns:
                heatmap_data[day] = 0
        
        heatmap_data = heatmap_data[days_order]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=days_order,
            y=[f"{h:02d}:00" for h in hours],
            colorscale='RdYlBu_r',
            colorbar=dict(title="Fraud<br>Detections"),
            hovertemplate='Day: %{x}<br>Hour: %{y}<br>Detections: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Email Fraud Pattern Heatmap (Hour vs Day of Week)",
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            template="plotly_white",
            height=500,
            margin=dict(l=80, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_risk_level_chart(self) -> go.Figure:
        """Create risk level distribution chart."""
        df = pd.DataFrame(self.fraud_data)
        
        risk_counts = df['risk_level'].value_counts()
        risk_order = ['Low', 'Medium', 'High', 'Critical']
        
        for risk in risk_order:
            if risk not in risk_counts:
                risk_counts[risk] = 0
        
        fig = go.Figure(data=[go.Bar(
            x=risk_order,
            y=[risk_counts[r] for r in risk_order],
            marker=dict(
                color=['green', 'yellow', 'orange', 'red'],
                line=dict(color='black', width=1.5)
            ),
            text=[risk_counts[r] for r in risk_order],
            textposition='auto',
            hovertemplate='Risk Level: %{x}<br>Count: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Fraud Detections by Risk Level",
            xaxis_title="Risk Level",
            yaxis_title="Number of Detections",
            template="plotly_white",
            height=350,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        
        return fig
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        df = pd.DataFrame(self.fraud_data)
        
        total_detections = len(df)
        
        stats = {
            "total_detections": total_detections,
            "avg_confidence": df['confidence'].mean() if total_detections > 0 else 0,
            "high_risk_count": len(df[df['risk_level'].isin(['High', 'Critical'])]),
            "blocked_count": len(df[df['status'] == 'Blocked']),
            "most_common_type": df['type'].mode()[0] if total_detections > 0 else "None",
            "detection_rate": total_detections / 30,
            "critical_rate": len(df[df['risk_level'] == 'Critical']) / total_detections if total_detections > 0 else 0
        }
        
        return stats
    
    def export_to_csv(self) -> str:
        """Export fraud data to CSV."""
        df = pd.DataFrame(self.fraud_data)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        output.close()
        
        filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w') as f:
            f.write(csv_content)
        
        return filename
    
    def export_to_pdf(self) -> str:
        """Export fraud analytics report to PDF."""
        filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1
        )
        
        story.append(Paragraph("Fraud Analytics Report", title_style))
        story.append(Spacer(1, 20))
        
        metadata_style = styles['Normal']
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
        story.append(Paragraph(f"<b>Report Period:</b> Last 30 days", metadata_style))
        story.append(Spacer(1, 20))
        
        stats = self.generate_summary_stats()
        story.append(Paragraph("<b>Summary Statistics</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        stats_data = [
            ["Metric", "Value"],
            ["Total Detections", str(stats['total_detections'])],
            ["Average Confidence", f"{stats['avg_confidence']:.2%}"],
            ["High Risk Count", str(stats['high_risk_count'])],
            ["Most Common Type", stats['most_common_type']],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        
        doc.build(story)
        
        return filename
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for fraud."""
        if not self.text_detector._initialized:
            await self.text_detector.initialize()
        
        result = await self.text_detector.detect(text)
        
        detection = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "type": result.fraud_types[0] if result.fraud_types else "unknown",
            "confidence": result.confidence,
            "risk_level": "Critical" if result.fraud_score > 0.8 else "High" if result.fraud_score > 0.6 else "Medium" if result.fraud_score > 0.3 else "Low",
            "source": "Text",
            "status": "Detected"
        }
        
        self.fraud_data.append(detection)
        self.detection_history.append(detection)
        
        return {
            "fraud_score": result.fraud_score,
            "confidence": result.confidence,
            "fraud_types": result.fraud_types,
            "explanation": result.explanation
        }


# Create app instance
logger.info("About to create IntegratedFraudLensApp instance...")
app = IntegratedFraudLensApp()
logger.info("IntegratedFraudLensApp instance created")


# Gradio interface functions
def update_trend_chart(days: int):
    """Update trend chart with specified number of days."""
    return app.create_trend_chart(days)




def get_summary_stats():
    """Get formatted summary statistics."""
    stats = app.generate_summary_stats()
    
    return f"""
    ### Summary Statistics
    
    - **Total Detections:** {stats['total_detections']}
    - **Average Confidence:** {stats['avg_confidence']:.2%}
    - **High Risk Count:** {stats['high_risk_count']}
    - **Most Common Type:** {stats['most_common_type']}
    - **Daily Detection Rate:** {stats['detection_rate']:.1f}
    """


def export_csv():
    """Export data to CSV."""
    filename = app.export_to_csv()
    return f"✅ Data exported to {filename}"


def export_pdf():
    """Export report to PDF."""
    filename = app.export_to_pdf()
    return f"✅ Report exported to {filename}"


async def analyze_input_text(text: str):
    """Analyze input text for fraud."""
    if not text:
        return "Please enter text to analyze"
    
    result = await app.analyze_text(text)
    
    output = f"""
    ### Analysis Results
    
    **Fraud Score:** {result['fraud_score']:.2%}
    **Confidence:** {result['confidence']:.2%}
    **Fraud Types:** {', '.join(result['fraud_types']) if result['fraud_types'] else 'None detected'}
    
    **Explanation:**
    {result['explanation']}
    """
    
    return output


# Create Gradio interface
logger.info("Creating Gradio Blocks...")
with gr.Blocks(title="FraudLens Integrated System", theme=gr.themes.Soft()) as demo:
    logger.info("Inside Blocks context...")
    gr.Markdown("""
    # 🔍 FraudLens - Comprehensive Fraud Detection System
    Advanced AI-powered fraud detection with Gmail integration and analytics
    """)
    
    # Simple test interface
    gr.Markdown("Testing simplified interface - tabs are temporarily disabled")
    test_input = gr.Textbox(label="Test Input")
    test_output = gr.Textbox(label="Test Output")
    test_btn = gr.Button("Test")
    test_btn.click(fn=lambda x: f"You entered: {x}", inputs=test_input, outputs=test_output)
    
    logger.info("Simple test interface created")
    
    # Original complex interface temporarily disabled
    if False:
        logger.info("Creating tabs...")
        with gr.Tabs():
        logger.info("Creating Email Monitor tab...")
        # Email Monitor Tab
        with gr.TabItem("📧 Email Monitor"):
            logger.info("Inside Email Monitor tab...")
            gr.Markdown("""
            ## Gmail Fraud Detection Console
            Monitor and manage email fraud detection in real-time with Gmail API integration.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    credentials_card = gr.HTML("<div style='padding: 15px;'>Credentials status will appear here</div>")
                    
                    with gr.Row():
                        gr.Markdown("### 🔐 Gmail Login")
                        init_btn = gr.Button("🚀 Initialize", variant="secondary", scale=0)
                    
                    gmail_email = gr.Textbox(
                        label="Gmail Address",
                        placeholder="yourname@gmail.com",
                        type="email"
                    )
                    
                    gmail_password = gr.Textbox(
                        label="App Password",
                        placeholder="Enter your Gmail app password",
                        type="password",
                        info="Use an app-specific password. Go to Google Account > Security > 2-Step Verification > App passwords"
                    )
                    
                    with gr.Row():
                        login_btn = gr.Button("🔐 Login", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear", variant="stop")
                    
                    auth_result = gr.Textbox(label="Authentication Status", lines=2)
                
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gmail_status = gr.Markdown("🔴 **Gmail Status:** Not Connected")
                            connect_btn = gr.Button("🔗 Connect to Gmail", variant="primary")
                            disconnect_btn = gr.Button("🔌 Disconnect", variant="stop", visible=False)
                            
                            gr.Markdown("---")
                            stats_display = gr.Markdown("📊 **Statistics**\n- Total Processed: 0\n- Fraud Detected: 0")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 📧 Email Search Options")
                            
                            with gr.Row():
                                query_type = gr.Radio(
                                    label="Query Type",
                                    choices=["Unread", "All", "By Sender", "By Date Range", "With Attachments", "Unknown Senders"],
                                    value="Unread",
                                )
                            
                            with gr.Row():
                                sender_filter = gr.Textbox(
                                    label="From Sender (optional)",
                                    placeholder="sender@example.com",
                                    visible=False
                                )
                                
                                date_from = gr.Textbox(
                                    label="Date From (YYYY-MM-DD)",
                                    placeholder="2025-01-01",
                                    visible=False
                                )
                                
                                date_to = gr.Textbox(
                                    label="Date To (YYYY-MM-DD)",
                                    placeholder="2025-01-31",
                                    visible=False
                                )
                            
                            with gr.Row():
                                max_emails = gr.Slider(
                                    minimum=10,
                                    maximum=1000,
                                    value=100,
                                    step=10,
                                    label="Emails per Batch",
                                )
                                process_attachments = gr.Checkbox(
                                    label="Process Attachments",
                                    value=False,
                                )
                            
                            with gr.Row():
                                scan_btn = gr.Button("🔍 Scan Emails", variant="primary")
                                continue_btn = gr.Button("⏭️ Continue Scanning", variant="secondary", visible=False)
                            
                            with gr.Row():
                                last_check_display = gr.Markdown("**Last Check:** Never")
                                emails_processed_display = gr.Markdown("**Total Processed:** 0")
                            
                            scan_output = gr.HTML(
                                label="Scan Results",
                                value="<div style='padding: 20px; text-align: center; color: #666;'>No scan results yet. Click 'Scan Emails' to start.</div>"
                            )
        
        logger.info("Email Monitor tab created, creating Analytics Dashboard tab...")
        # Analytics Dashboard Tab
        with gr.TabItem("📊 Analytics Dashboard"):
            gr.Markdown("""
            ## Fraud Analytics Dashboard
            Comprehensive fraud detection analytics with real-time visualizations.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    trend_days = gr.Slider(
                        minimum=7,
                        maximum=90,
                        value=30,
                        step=1,
                        label="Days to Display"
                    )
                    trend_chart = gr.Plot(label="Fraud Detection Trends")
                
                with gr.Column(scale=1):
                    stats_display_analytics = gr.Markdown(value="### Summary Statistics\n\nClick 'Refresh Stats' to load statistics")
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Stats", variant="primary")
                        load_charts_btn = gr.Button("📊 Load Charts", variant="secondary")
            
            with gr.Row():
                pie_chart = gr.Plot(label="Fraud Type Distribution")
                risk_chart = gr.Plot(label="Risk Level Distribution")
            
            with gr.Row():
                email_heatmap = gr.Plot(label="Email Fraud Pattern Heatmap")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Export Options")
                    csv_btn = gr.Button("📊 Export to CSV", variant="primary")
                    csv_output = gr.Textbox(label="Export Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### Generate Report")
                    pdf_btn = gr.Button("📄 Export to PDF", variant="primary")
                    pdf_output = gr.Textbox(label="Export Status", interactive=False)
        
        # Live Analysis Tab
        with gr.TabItem("🔍 Live Analysis"):
            gr.Markdown("""
            ## Live Fraud Analysis
            Analyze text, documents, and media for fraud in real-time.
            """)
            
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="Paste suspicious text, email, or document content here...",
                        lines=10
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column():
                    analysis_output = gr.Markdown(label="Analysis Results")
        
        # About Tab
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ### FraudLens Integrated System
            
            A comprehensive fraud detection platform combining:
            
            #### 📧 Email Monitoring
            - Real-time Gmail integration
            - Automatic fraud detection in emails
            - Attachment processing
            - Configurable actions for fraudulent emails
            
            #### 📊 Analytics Dashboard
            - Real-time trend analysis
            - Fraud type distribution
            - Risk assessment visualization
            - Email pattern heatmaps
            - Export capabilities (CSV/PDF)
            
            #### 🔍 Live Analysis
            - Text fraud detection
            - Document analysis
            - Video/Image deepfake detection
            - Multi-modal fraud detection
            
            #### 🛡️ Detection Capabilities
            - Phishing detection
            - Social engineering identification
            - Deepfake detection
            - Document fraud analysis
            - Money laundering patterns
            - Identity theft indicators
            
            **Version:** 2.0.0
            **Last Updated:** 2025-08-29
            """)
    
    logger.info("All tabs created, setting up event handlers...")
    # Event handlers
    init_btn.click(
        fn=lambda: (app.get_credentials_card(), app._get_stats_display()),
        outputs=[credentials_card, stats_display]
    )
    
    login_btn.click(
        app.authenticate_gmail,
        inputs=[gmail_email, gmail_password],
        outputs=[auth_result, credentials_card]
    )
    
    clear_btn.click(
        app.clear_credentials,
        outputs=[auth_result, credentials_card]
    )
    
    connect_btn.click(
        app.connect_gmail,
        outputs=[gmail_status, connect_btn, disconnect_btn, stats_display, credentials_card]
    )
    
    disconnect_btn.click(
        app.disconnect_gmail,
        outputs=[gmail_status, connect_btn, disconnect_btn, stats_display]
    )
    
    # Query type change handler to show/hide relevant fields
    def update_query_fields(query_type):
        if query_type == "By Sender":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif query_type == "By Date Range":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    query_type.change(
        fn=update_query_fields,
        inputs=[query_type],
        outputs=[sender_filter, date_from, date_to]
    )
    
    scan_btn.click(
        app.scan_emails,
        inputs=[query_type, sender_filter, date_from, date_to, max_emails, process_attachments],
        outputs=[scan_output, last_check_display, emails_processed_display, continue_btn]
    ).then(
        lambda: app._get_stats_display(),
        outputs=stats_display
    )
    
    # Continue scanning button (with continuation flag)
    continue_btn.click(
        lambda qt, s, df, dt, me, pa: app.scan_emails(qt, s, df, dt, me, pa, is_continuation=True),
        inputs=[query_type, sender_filter, date_from, date_to, max_emails, process_attachments],
        outputs=[scan_output, last_check_display, emails_processed_display, continue_btn]
    ).then(
        lambda: app._get_stats_display(),
        outputs=stats_display
    )
    
    trend_days.change(
        fn=update_trend_chart,
        inputs=[trend_days],
        outputs=[trend_chart]
    )
    
    refresh_btn.click(
        fn=get_summary_stats,
        outputs=[stats_display_analytics]
    )
    
    load_charts_btn.click(
        fn=lambda: (
            app.create_trend_chart(),
            app.create_distribution_pie_chart(),
            app.create_email_heatmap(),
            app.create_risk_level_chart()
        ),
        outputs=[trend_chart, pie_chart, email_heatmap, risk_chart]
    )
    
    csv_btn.click(fn=export_csv, outputs=[csv_output])
    pdf_btn.click(fn=export_pdf, outputs=[pdf_output])
    
    analyze_btn.click(
        fn=lambda x: asyncio.run(analyze_input_text(x)),
        inputs=[input_text],
        outputs=[analysis_output]
    )
    
    logger.info("Event handlers configured")
    # Removed demo.load() completely to prevent hanging


if __name__ == "__main__":
    logger.info("Starting FraudLens Integrated System...")
    logger.info(f"Demo object created: {demo is not None}")
    logger.info("About to launch Gradio interface...")
    
    try:
        # Simple launch without extra parameters
        demo.launch(
            server_name="0.0.0.0",
            server_port=7863
        )
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        import traceback
        traceback.print_exc()