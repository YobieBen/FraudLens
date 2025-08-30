#!/usr/bin/env python3
"""
Fraud Analytics Dashboard with Gradio
Provides comprehensive fraud analysis visualization and reporting
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import io
import base64
from collections import defaultdict, Counter
import random
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fraudlens.processors.text.detector import TextFraudDetector
# from fraudlens.api.gmail_integration import GmailFraudDetector  # Optional import


class FraudAnalyticsDashboard:
    """Advanced fraud analytics dashboard with visualizations."""
    
    def __init__(self):
        """Initialize the analytics dashboard."""
        self.detector = TextFraudDetector()
        self.gmail_detector = None
        self.fraud_data = []
        self.email_data = []
        self.detection_history = []
        
        # Initialize with sample data for demo
        self._generate_sample_data()
        
    def _generate_sample_data(self):
        """Generate sample fraud data for demonstration."""
        # Generate 30 days of sample data
        end_date = datetime.now()
        
        fraud_types = ["phishing", "deepfake", "identity_theft", "social_engineering", 
                      "document_fraud", "money_laundering", "scam"]
        
        for i in range(30):
            date = end_date - timedelta(days=i)
            
            # Generate random fraud detections for each day
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
        
        # Generate email-specific fraud patterns
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
    
    def create_trend_chart(self, days: int = 30) -> go.Figure:
        """Create fraud trends over time chart."""
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.fraud_data)
        
        # Get data for specified number of days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= start_date]
        
        # Group by date and type
        daily_counts = df.groupby(['date', 'type']).size().unstack(fill_value=0)
        
        # Create line chart
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
        
        # Count fraud types
        type_counts = df['type'].value_counts()
        
        # Create pie chart
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
        
        # Create hour x day matrix for fraud detection
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        
        # Count fraud detections by hour and day
        fraud_df = df[df['fraud_detected'] == True]
        
        # Create pivot table
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
        
        # Ensure all days are present
        for day in days_order:
            if day not in heatmap_data.columns:
                heatmap_data[day] = 0
        
        heatmap_data = heatmap_data[days_order]
        
        # Create heatmap
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
        
        # Count by risk level
        risk_counts = df['risk_level'].value_counts()
        risk_order = ['Low', 'Medium', 'High', 'Critical']
        
        # Ensure all risk levels are present
        for risk in risk_order:
            if risk not in risk_counts:
                risk_counts[risk] = 0
        
        # Create bar chart
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
    
    def create_source_distribution_chart(self) -> go.Figure:
        """Create fraud source distribution chart."""
        df = pd.DataFrame(self.fraud_data)
        
        # Count by source
        source_counts = df['source'].value_counts()
        
        # Create horizontal bar chart
        fig = go.Figure(data=[go.Bar(
            y=source_counts.index,
            x=source_counts.values,
            orientation='h',
            marker=dict(
                color=px.colors.qualitative.Pastel,
                line=dict(color='black', width=1)
            ),
            text=source_counts.values,
            textposition='auto',
            hovertemplate='Source: %{y}<br>Count: %{x}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Fraud Detections by Source",
            xaxis_title="Number of Detections",
            yaxis_title="Source",
            template="plotly_white",
            height=350,
            margin=dict(l=100, r=50, t=80, b=50),
            showlegend=False
        )
        
        return fig
    
    def create_confidence_histogram(self) -> go.Figure:
        """Create confidence score histogram."""
        df = pd.DataFrame(self.fraud_data)
        
        # Create histogram
        fig = go.Figure(data=[go.Histogram(
            x=df['confidence'],
            nbinsx=20,
            marker=dict(
                color='rgba(100, 150, 250, 0.7)',
                line=dict(color='black', width=1)
            ),
            hovertemplate='Confidence Range: %{x}<br>Count: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Detection Confidence Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
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
        
        # Calculate statistics
        stats = {
            "total_detections": total_detections,
            "avg_confidence": df['confidence'].mean() if total_detections > 0 else 0,
            "high_risk_count": len(df[df['risk_level'].isin(['High', 'Critical'])]),
            "blocked_count": len(df[df['status'] == 'Blocked']),
            "most_common_type": df['type'].mode()[0] if total_detections > 0 else "None",
            "detection_rate": total_detections / 30,  # Per day average
            "critical_rate": len(df[df['risk_level'] == 'Critical']) / total_detections if total_detections > 0 else 0
        }
        
        return stats
    
    def export_to_csv(self) -> str:
        """Export fraud data to CSV."""
        df = pd.DataFrame(self.fraud_data)
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        # Get CSV content
        csv_content = output.getvalue()
        output.close()
        
        # Save to file
        filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w') as f:
            f.write(csv_content)
        
        return filename
    
    def export_to_pdf(self) -> str:
        """Export fraud analytics report to PDF."""
        filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        story.append(Paragraph("Fraud Analytics Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        metadata_style = styles['Normal']
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
        story.append(Paragraph(f"<b>Report Period:</b> Last 30 days", metadata_style))
        story.append(Spacer(1, 20))
        
        # Summary statistics
        stats = self.generate_summary_stats()
        story.append(Paragraph("<b>Summary Statistics</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        stats_data = [
            ["Metric", "Value"],
            ["Total Detections", str(stats['total_detections'])],
            ["Average Confidence", f"{stats['avg_confidence']:.2%}"],
            ["High Risk Count", str(stats['high_risk_count'])],
            ["Blocked Count", str(stats['blocked_count'])],
            ["Most Common Type", stats['most_common_type']],
            ["Daily Detection Rate", f"{stats['detection_rate']:.1f}"],
            ["Critical Rate", f"{stats['critical_rate']:.2%}"]
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
        story.append(PageBreak())
        
        # Top fraud types
        story.append(Paragraph("<b>Top Fraud Types</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        df = pd.DataFrame(self.fraud_data)
        top_types = df['type'].value_counts().head(5)
        
        type_data = [["Fraud Type", "Count", "Percentage"]]
        total = len(df)
        for fraud_type, count in top_types.items():
            type_data.append([
                fraud_type.replace('_', ' ').title(),
                str(count),
                f"{(count/total)*100:.1f}%"
            ])
        
        type_table = Table(type_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        type_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(type_table)
        story.append(Spacer(1, 20))
        
        # Risk distribution
        story.append(Paragraph("<b>Risk Level Distribution</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        risk_counts = df['risk_level'].value_counts()
        risk_data = [["Risk Level", "Count", "Percentage"]]
        
        for risk in ['Critical', 'High', 'Medium', 'Low']:
            if risk in risk_counts:
                count = risk_counts[risk]
                risk_data.append([risk, str(count), f"{(count/total)*100:.1f}%"])
        
        risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(risk_table)
        
        # Build PDF
        doc.build(story)
        
        return filename
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for fraud and update dashboard."""
        if not self.detector._initialized:
            await self.detector.initialize()
        
        result = await self.detector.detect(text)
        
        # Add to fraud data
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


# Create dashboard instance
dashboard = FraudAnalyticsDashboard()


# Gradio interface functions
def update_trend_chart(days: int):
    """Update trend chart with specified number of days."""
    return dashboard.create_trend_chart(days)


def update_all_charts():
    """Update all dashboard charts."""
    return (
        dashboard.create_trend_chart(),
        dashboard.create_distribution_pie_chart(),
        dashboard.create_email_heatmap(),
        dashboard.create_risk_level_chart(),
        dashboard.create_source_distribution_chart(),
        dashboard.create_confidence_histogram()
    )


def get_summary_stats():
    """Get formatted summary statistics."""
    stats = dashboard.generate_summary_stats()
    
    return f"""
    ### Summary Statistics
    
    - **Total Detections:** {stats['total_detections']}
    - **Average Confidence:** {stats['avg_confidence']:.2%}
    - **High Risk Count:** {stats['high_risk_count']}
    - **Blocked Count:** {stats['blocked_count']}
    - **Most Common Type:** {stats['most_common_type']}
    - **Daily Detection Rate:** {stats['detection_rate']:.1f}
    - **Critical Rate:** {stats['critical_rate']:.2%}
    """


def export_csv():
    """Export data to CSV."""
    filename = dashboard.export_to_csv()
    return f"✅ Data exported to {filename}"


def export_pdf():
    """Export report to PDF."""
    filename = dashboard.export_to_pdf()
    return f"✅ Report exported to {filename}"


async def analyze_input_text(text: str):
    """Analyze input text for fraud."""
    if not text:
        return "Please enter text to analyze"
    
    result = await dashboard.analyze_text(text)
    
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
with gr.Blocks(title="FraudLens Analytics Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 FraudLens Analytics Dashboard
    
    Comprehensive fraud detection analytics with real-time visualizations and reporting.
    """)
    
    with gr.Tab("📈 Overview"):
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
                stats_display = gr.Markdown(value=get_summary_stats())
                refresh_btn = gr.Button("🔄 Refresh Stats", variant="primary")
        
        with gr.Row():
            pie_chart = gr.Plot(label="Fraud Type Distribution")
            risk_chart = gr.Plot(label="Risk Level Distribution")
        
        trend_days.change(
            fn=update_trend_chart,
            inputs=[trend_days],
            outputs=[trend_chart]
        )
        
        refresh_btn.click(
            fn=get_summary_stats,
            outputs=[stats_display]
        )
    
    with gr.Tab("📧 Email Analysis"):
        with gr.Row():
            email_heatmap = gr.Plot(label="Email Fraud Pattern Heatmap")
        
        with gr.Row():
            source_chart = gr.Plot(label="Fraud by Source")
            confidence_hist = gr.Plot(label="Confidence Distribution")
    
    with gr.Tab("🔍 Live Analysis"):
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
        
        analyze_btn.click(
            fn=lambda x: asyncio.run(analyze_input_text(x)),
            inputs=[input_text],
            outputs=[analysis_output]
        )
    
    with gr.Tab("📤 Export"):
        gr.Markdown("""
        ### Export Options
        
        Export your fraud analytics data and reports in various formats.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                #### CSV Export
                Export raw fraud detection data to CSV format for further analysis.
                """)
                csv_btn = gr.Button("📊 Export to CSV", variant="primary")
                csv_output = gr.Textbox(label="Export Status", interactive=False)
            
            with gr.Column():
                gr.Markdown("""
                #### PDF Report
                Generate a comprehensive PDF report with charts and statistics.
                """)
                pdf_btn = gr.Button("📄 Export to PDF", variant="primary")
                pdf_output = gr.Textbox(label="Export Status", interactive=False)
        
        csv_btn.click(fn=export_csv, outputs=[csv_output])
        pdf_btn.click(fn=export_pdf, outputs=[pdf_output])
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ### FraudLens Analytics Dashboard
        
        This dashboard provides comprehensive fraud detection analytics including:
        
        - **Real-time Trend Analysis**: Monitor fraud patterns over time
        - **Type Distribution**: Understand the breakdown of fraud types
        - **Risk Assessment**: View risk level distributions
        - **Email Pattern Analysis**: Identify temporal patterns in email fraud
        - **Confidence Metrics**: Analyze detection confidence levels
        - **Export Capabilities**: Generate CSV data exports and PDF reports
        
        #### Features:
        - 📊 Interactive charts powered by Plotly
        - 🔄 Real-time data updates
        - 📧 Email fraud pattern heatmaps
        - 📈 Trend analysis with customizable date ranges
        - 📤 Export to CSV and PDF formats
        - 🔍 Live text analysis
        
        #### Data Sources:
        - Email monitoring via Gmail API
        - Text analysis using advanced NLP
        - Video/Image deepfake detection
        - Document fraud analysis
        
        **Version:** 1.0.0
        **Last Updated:** 2025-08-29
        """)
    
    # Load initial charts
    demo.load(
        fn=update_all_charts,
        outputs=[trend_chart, pie_chart, email_heatmap, risk_chart, source_chart, confidence_hist]
    )


if __name__ == "__main__":
    logger.info("Starting FraudLens Analytics Dashboard...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        debug=True
    )