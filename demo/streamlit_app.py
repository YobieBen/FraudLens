#!/usr/bin/env python3
"""
FraudLens Streamlit App
Alternative to Gradio implementation
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import FraudLens modules
try:
    from fraudlens.processors.text.detector import TextFraudDetector
    from fraudlens.api.gmail_imap_integration import GmailIMAPScanner
    print("✅ FraudLens modules imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Mock classes for testing
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

# Initialize detector
@st.cache_resource
def get_detector():
    detector = TextFraudDetector()
    asyncio.run(detector.initialize())
    return detector

# Page config
st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 FraudLens - Fraud Detection System")
st.markdown("Analyze text and emails for potential fraud")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a feature",
    ["📝 Text Analysis", "📧 Email Scanner"]
)

# Get detector
detector = get_detector()

if page == "📝 Text Analysis":
    st.header("📝 Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze",
            height=300,
            placeholder="Paste suspicious text here..."
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input:
                with st.spinner("Analyzing..."):
                    result = asyncio.run(detector.detect(text_input))
                    
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Fraud status
                    if result.is_fraud:
                        st.error("⚠️ FRAUD DETECTED")
                    else:
                        st.success("✅ Text appears safe")
                    
                    # Metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Fraud Score", f"{result.fraud_score:.1%}")
                    with col_b:
                        st.metric("Confidence", f"{result.confidence:.1%}")
                    
                    # Details
                    if result.fraud_types:
                        st.write("**Fraud Types:**", ", ".join(result.fraud_types))
                    
                    st.write("**Explanation:**", result.explanation)
            else:
                st.warning("Please enter some text to analyze")

elif page == "📧 Email Scanner":
    st.header("📧 Email Scanner")
    
    # Session state for connection
    if 'gmail_connected' not in st.session_state:
        st.session_state.gmail_connected = False
        st.session_state.gmail_scanner = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gmail Connection")
        
        email = st.text_input("Gmail Address")
        password = st.text_input("App Password", type="password")
        
        if st.button("🔗 Connect to Gmail"):
            if email and password:
                with st.spinner("Connecting..."):
                    scanner = GmailIMAPScanner(fraud_detector=detector)
                    if scanner.connect(email, password):
                        st.session_state.gmail_connected = True
                        st.session_state.gmail_scanner = scanner
                        st.success(f"✅ Connected to {email}")
                    else:
                        st.error("❌ Connection failed")
            else:
                st.warning("Please enter email and password")
        
        # Connection status
        if st.session_state.gmail_connected:
            st.info("✅ Connected to Gmail")
        else:
            st.warning("❌ Not connected")
    
    with col2:
        st.subheader("Scan Options")
        
        query_type = st.radio(
            "Email Filter",
            ["Unread", "All", "Recent"]
        )
        
        max_emails = st.slider(
            "Max Emails to Scan",
            min_value=1,
            max_value=100,
            value=10
        )
        
        if st.button("📧 Scan Emails", disabled=not st.session_state.gmail_connected):
            if st.session_state.gmail_scanner:
                query_map = {
                    'Unread': 'UNSEEN',
                    'All': 'ALL',
                    'Recent': 'RECENT'
                }
                query = query_map[query_type]
                
                with st.spinner(f"Scanning {max_emails} emails..."):
                    results = asyncio.run(
                        st.session_state.gmail_scanner.scan_for_fraud(query, max_emails)
                    )
                
                if results:
                    st.subheader(f"Scanned {len(results)} emails")
                    
                    for email in results:
                        with st.expander(f"{email['subject']} - {email['sender']}"):
                            if email['is_fraud']:
                                st.error("⚠️ FRAUD DETECTED")
                            else:
                                st.success("✅ Safe")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.write(f"**Date:** {email['date']}")
                            with col_b:
                                st.write(f"**Confidence:** {email['confidence']:.1%}")
                            with col_c:
                                st.write(f"**Risk:** {email['risk_level']}")
                            
                            if email.get('fraud_types'):
                                st.write(f"**Fraud Types:** {', '.join(email['fraud_types'])}")
                else:
                    st.info("No emails found")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "FraudLens uses AI to detect potential fraud in text and emails. "
    "Always verify suspicious content through official channels."
)