#!/usr/bin/env python3
"""
FraudLens Analytics Dashboard
Comprehensive fraud analytics with visualizations and export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import csv
import io
import base64
from pathlib import Path
import asyncio
import random
from typing import Dict, List, Any, Optional

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="FraudLens Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
    }
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .export-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .fraud-alert {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-alert {
        background: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FraudAnalyticsDashboard:
    """Comprehensive fraud analytics dashboard"""
    
    def __init__(self):
        """Initialize dashboard with sample data"""
        self.initialize_data()
        
    def initialize_data(self):
        """Initialize or load fraud data"""
        # Generate sample fraud data for demonstration
        np.random.seed(42)
        
        # Generate time series data for the last 90 days
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        # Fraud trends data
        self.fraud_trends = pd.DataFrame({
            'date': dates,
            'text_fraud': np.random.poisson(15, 90) + np.random.randint(-5, 5, 90),
            'image_fraud': np.random.poisson(8, 90) + np.random.randint(-3, 3, 90),
            'video_fraud': np.random.poisson(5, 90) + np.random.randint(-2, 2, 90),
            'document_fraud': np.random.poisson(10, 90) + np.random.randint(-4, 4, 90),
            'email_fraud': np.random.poisson(20, 90) + np.random.randint(-7, 7, 90),
        })
        
        # Ensure no negative values
        for col in self.fraud_trends.columns[1:]:
            self.fraud_trends[col] = self.fraud_trends[col].clip(lower=0)
        
        # Fraud type distribution
        self.fraud_types = {
            'Phishing': 35,
            'Identity Theft': 20,
            'Document Forgery': 15,
            'Deepfake': 12,
            'Financial Fraud': 10,
            'Social Engineering': 8
        }
        
        # Email fraud patterns (hourly heatmap data)
        hours = list(range(24))
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create heatmap data with realistic patterns (more fraud during business hours)
        self.email_heatmap_data = []
        for day in days:
            for hour in hours:
                # Higher fraud during business hours on weekdays
                if day in ['Saturday', 'Sunday']:
                    base_rate = 5
                else:
                    if 9 <= hour <= 17:
                        base_rate = 20
                    elif 6 <= hour <= 9 or 17 <= hour <= 20:
                        base_rate = 12
                    else:
                        base_rate = 3
                
                fraud_count = np.random.poisson(base_rate) + np.random.randint(-2, 3)
                self.email_heatmap_data.append({
                    'Day': day,
                    'Hour': hour,
                    'Fraud_Count': max(0, fraud_count)
                })
        
        self.email_heatmap_df = pd.DataFrame(self.email_heatmap_data)
        
        # Risk scores distribution
        self.risk_scores = {
            'Critical': 15,
            'High': 25,
            'Medium': 35,
            'Low': 25
        }
        
        # Geographic distribution (sample data)
        self.geographic_data = pd.DataFrame({
            'Country': ['United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 
                       'France', 'Japan', 'Brazil', 'India', 'China'],
            'Fraud_Cases': [450, 280, 190, 150, 210, 180, 160, 220, 380, 420],
            'Latitude': [37.0902, 51.5074, 56.1304, -25.2744, 51.1657, 
                        46.2276, 36.2048, -14.2350, 20.5937, 35.8617],
            'Longitude': [-95.7129, -0.1278, -106.3468, 133.7751, 10.4515,
                         2.2137, 138.2529, -51.9253, 78.9629, 104.1954]
        })
        
        # Detection accuracy metrics
        self.accuracy_metrics = {
            'Text Detection': 96.5,
            'Image Detection': 93.2,
            'Video Detection': 89.7,
            'Document Validation': 94.8,
            'Email Scanning': 97.3
        }
        
        # Recent alerts
        self.recent_alerts = self._generate_recent_alerts()
    
    def _generate_recent_alerts(self):
        """Generate recent fraud alerts"""
        alert_types = ['Phishing Email', 'Fake Document', 'Deepfake Video', 'Fraudulent Text', 'Manipulated Image']
        severities = ['Critical', 'High', 'Medium', 'Low']
        
        alerts = []
        for i in range(20):
            time_ago = datetime.now() - timedelta(minutes=np.random.randint(1, 1440))
            alerts.append({
                'Time': time_ago.strftime('%Y-%m-%d %H:%M'),
                'Type': np.random.choice(alert_types),
                'Severity': np.random.choice(severities, p=[0.1, 0.3, 0.4, 0.2]),
                'Confidence': f"{np.random.uniform(75, 99):.1f}%",
                'Status': np.random.choice(['Blocked', 'Flagged', 'Under Review'], p=[0.5, 0.3, 0.2])
            })
        
        return pd.DataFrame(alerts)
    
    def create_fraud_trends_chart(self):
        """Create fraud trends over time chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Fraud Detection Trends', 'Cumulative Fraud Cases'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Daily trends
        for col in self.fraud_trends.columns[1:]:
            fig.add_trace(
                go.Scatter(
                    x=self.fraud_trends['date'],
                    y=self.fraud_trends[col],
                    name=col.replace('_', ' ').title(),
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate='%{y} cases<br>%{x|%B %d, %Y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Cumulative trends
        cumulative = self.fraud_trends[self.fraud_trends.columns[1:]].cumsum()
        cumulative['date'] = self.fraud_trends['date']
        
        for col in cumulative.columns[:-1]:
            fig.add_trace(
                go.Scatter(
                    x=cumulative['date'],
                    y=cumulative[col],
                    name=col.replace('_', ' ').title(),
                    mode='lines',
                    fill='tonexty',
                    stackgroup='one',
                    hovertemplate='%{y} total cases<br>%{x|%B %d, %Y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified',
            title_text="Fraud Detection Trends Analysis",
            title_font_size=20
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Daily Cases", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Cases", row=2, col=1)
        
        return fig
    
    def create_fraud_distribution_pie(self):
        """Create fraud type distribution pie chart"""
        fig = go.Figure(data=[
            go.Pie(
                labels=list(self.fraud_types.keys()),
                values=list(self.fraud_types.values()),
                hole=0.4,
                marker=dict(
                    colors=px.colors.sequential.Viridis,
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>%{value} cases<br>%{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Fraud Type Distribution",
            title_font_size=20,
            height=500,
            annotations=[
                dict(
                    text='Fraud<br>Types',
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    def create_email_fraud_heatmap(self):
        """Create email fraud patterns heatmap"""
        # Pivot data for heatmap
        heatmap_pivot = self.email_heatmap_df.pivot(
            index='Day',
            columns='Hour',
            values='Fraud_Count'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
            y=heatmap_pivot.index,
            colorscale='RdYlBu_r',
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Fraud Cases: %{z}<extra></extra>',
            colorbar=dict(title="Fraud Cases")
        ))
        
        fig.update_layout(
            title="Email Fraud Patterns - Weekly Heatmap",
            title_font_size=20,
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 24, 2)),
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
            )
        )
        
        return fig
    
    def create_risk_gauge(self, current_risk=72):
        """Create risk level gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_risk,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current Risk Level", 'font': {'size': 24}},
            delta={'reference': 60, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': 'lightgreen'},
                    {'range': [25, 50], 'color': 'yellow'},
                    {'range': [50, 75], 'color': 'orange'},
                    {'range': [75, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_geographic_map(self):
        """Create geographic distribution map"""
        fig = px.scatter_geo(
            self.geographic_data,
            lat='Latitude',
            lon='Longitude',
            size='Fraud_Cases',
            hover_name='Country',
            hover_data={'Fraud_Cases': True, 'Latitude': False, 'Longitude': False},
            color='Fraud_Cases',
            color_continuous_scale='Reds',
            size_max=50,
            title="Global Fraud Distribution"
        )
        
        fig.update_layout(
            height=500,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            )
        )
        
        return fig
    
    def create_accuracy_bar_chart(self):
        """Create detection accuracy bar chart"""
        fig = go.Figure(data=[
            go.Bar(
                x=list(self.accuracy_metrics.keys()),
                y=list(self.accuracy_metrics.values()),
                text=[f"{v:.1f}%" for v in self.accuracy_metrics.values()],
                textposition='auto',
                marker_color='lightblue',
                hovertemplate='%{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Detection Accuracy by Type",
            xaxis_title="Detection Type",
            yaxis_title="Accuracy (%)",
            height=400,
            yaxis_range=[0, 100]
        )
        
        # Add target line
        fig.add_hline(y=95, line_dash="dash", line_color="green", 
                     annotation_text="Target: 95%")
        
        return fig
    
    def export_to_csv(self):
        """Export data to CSV"""
        # Combine all data
        export_data = {
            'Fraud Trends': self.fraud_trends,
            'Recent Alerts': self.recent_alerts,
            'Geographic Distribution': self.geographic_data
        }
        
        # Create CSV in memory
        output = io.StringIO()
        
        for sheet_name, df in export_data.items():
            output.write(f"\n=== {sheet_name} ===\n")
            df.to_csv(output, index=False)
        
        return output.getvalue()
    
    def export_to_pdf(self):
        """Export dashboard to PDF report"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.pdfgen import canvas
            
            # Create PDF in memory
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            
            # Container for the 'Flowable' objects
            elements = []
            
            # Define styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#667eea'),
                alignment=TA_CENTER,
                spaceAfter=30
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#764ba2'),
                spaceAfter=12
            )
            
            # Add title
            elements.append(Paragraph("FraudLens Analytics Report", title_style))
            elements.append(Spacer(1, 12))
            
            # Add generation date
            elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Summary statistics
            elements.append(Paragraph("Executive Summary", heading_style))
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Fraud Cases (90 days)', str(self.fraud_trends[self.fraud_trends.columns[1:]].sum().sum())],
                ['Average Daily Cases', f"{self.fraud_trends[self.fraud_trends.columns[1:]].mean().sum():.1f}"],
                ['Most Common Fraud Type', max(self.fraud_types, key=self.fraud_types.get)],
                ['Current Risk Level', '72%'],
                ['Detection Accuracy', f"{np.mean(list(self.accuracy_metrics.values())):.1f}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            elements.append(PageBreak())
            
            # Fraud type distribution
            elements.append(Paragraph("Fraud Type Distribution", heading_style))
            
            fraud_type_data = [['Fraud Type', 'Percentage']]
            total = sum(self.fraud_types.values())
            for fraud_type, count in self.fraud_types.items():
                fraud_type_data.append([fraud_type, f"{(count/total)*100:.1f}%"])
            
            fraud_table = Table(fraud_type_data, colWidths=[3*inch, 2*inch])
            fraud_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(fraud_table)
            elements.append(Spacer(1, 20))
            
            # Recent alerts
            elements.append(Paragraph("Recent Fraud Alerts (Top 10)", heading_style))
            
            alerts_data = [['Time', 'Type', 'Severity', 'Status']]
            for _, alert in self.recent_alerts.head(10).iterrows():
                alerts_data.append([
                    alert['Time'],
                    alert['Type'],
                    alert['Severity'],
                    alert['Status']
                ])
            
            alerts_table = Table(alerts_data, colWidths=[2*inch, 2*inch, 1.5*inch, 1.5*inch])
            alerts_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(alerts_table)
            
            # Build PDF
            doc.build(elements)
            
            # Get PDF value
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except ImportError:
            st.error("ReportLab is required for PDF export. Install it with: pip install reportlab")
            return None

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='text-align: center; margin: 0;'>📊 FraudLens Analytics Dashboard</h1>
        <p style='text-align: center; margin-top: 10px; opacity: 0.9;'>
            Comprehensive Fraud Detection Analytics & Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = FraudAnalyticsDashboard()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        
        # Date range selector
        date_range = st.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            dashboard.initialize_data()
            st.rerun()
        
        st.markdown("---")
        
        # Export section
        st.markdown("### 📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 CSV", use_container_width=True):
                csv_data = dashboard.export_to_csv()
                b64 = base64.b64encode(csv_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="fraud_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("📑 PDF", use_container_width=True):
                pdf_data = dashboard.export_to_pdf()
                if pdf_data:
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="fraud_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">Download PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        total_cases = dashboard.fraud_trends[dashboard.fraud_trends.columns[1:]].sum().sum()
        st.metric("Total Cases (90d)", f"{total_cases:,}")
        st.metric("Avg Daily", f"{total_cases/90:.1f}")
        st.metric("Detection Rate", "96.5%", "↑ 2.3%")
    
    # Main content - Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style='color: #667eea; margin: 0;'>Active Threats</h4>
            <h2 style='margin: 10px 0;'>247</h2>
            <p style='color: red; margin: 0;'>↑ 12% from yesterday</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style='color: #764ba2; margin: 0;'>Blocked Today</h4>
            <h2 style='margin: 10px 0;'>1,832</h2>
            <p style='color: green; margin: 0;'>↓ 5% from average</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style='color: #667eea; margin: 0;'>Risk Score</h4>
            <h2 style='margin: 10px 0;'>72/100</h2>
            <p style='color: orange; margin: 0;'>High Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4 style='color: #764ba2; margin: 0;'>Accuracy</h4>
            <h2 style='margin: 10px 0;'>96.5%</h2>
            <p style='color: green; margin: 0;'>Above Target</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Main Charts
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "🥧 Distribution", "🗺️ Heatmap", "🌍 Geographic", "📊 Performance"])
    
    with tab1:
        st.plotly_chart(dashboard.create_fraud_trends_chart(), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(dashboard.create_fraud_distribution_pie(), use_container_width=True)
        with col2:
            st.markdown("### Risk Distribution")
            for risk_level, percentage in dashboard.risk_scores.items():
                color = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}[risk_level]
                st.markdown(f"""
                <div style='margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>{risk_level}</span>
                        <span>{percentage}%</span>
                    </div>
                    <div style='background: #e0e0e0; height: 20px; border-radius: 10px;'>
                        <div style='background: {color}; width: {percentage}%; height: 100%; border-radius: 10px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.plotly_chart(dashboard.create_email_fraud_heatmap(), use_container_width=True)
        st.info("📧 Peak fraud activity detected during business hours (9 AM - 5 PM) on weekdays")
    
    with tab4:
        st.plotly_chart(dashboard.create_geographic_map(), use_container_width=True)
    
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(dashboard.create_accuracy_bar_chart(), use_container_width=True)
        with col2:
            st.plotly_chart(dashboard.create_risk_gauge(), use_container_width=True)
    
    # Row 3: Recent Alerts Table
    st.markdown("### 🚨 Recent Fraud Alerts")
    
    # Style the dataframe
    styled_df = dashboard.recent_alerts.style.apply(
        lambda x: ['background-color: #ffcccc' if v == 'Critical' 
                  else 'background-color: #ffe6cc' if v == 'High'
                  else 'background-color: #ffffcc' if v == 'Medium'
                  else 'background-color: #ccffcc' if v == 'Low'
                  else '' for v in x],
        subset=['Severity']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with col2:
        st.markdown("**Data Range:** Last 90 days")
    
    with col3:
        st.markdown("**Next Refresh:** Auto-refresh in 5 minutes")

if __name__ == "__main__":
    main()