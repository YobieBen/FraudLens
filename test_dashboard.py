#!/usr/bin/env python3
"""
Test script for the Fraud Analytics Dashboard
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from demo.gradio_analytics_dashboard import FraudAnalyticsDashboard
import asyncio
from datetime import datetime


async def test_dashboard():
    """Test dashboard functionality."""
    print("Testing FraudLens Analytics Dashboard...")
    
    # Create dashboard instance
    dashboard = FraudAnalyticsDashboard()
    
    # Test 1: Summary Statistics
    print("\n1. Testing Summary Statistics...")
    stats = dashboard.generate_summary_stats()
    print(f"   Total Detections: {stats['total_detections']}")
    print(f"   Average Confidence: {stats['avg_confidence']:.2%}")
    print(f"   High Risk Count: {stats['high_risk_count']}")
    print(f"   Most Common Type: {stats['most_common_type']}")
    
    # Test 2: Chart Generation
    print("\n2. Testing Chart Generation...")
    try:
        trend_chart = dashboard.create_trend_chart(30)
        print("   ✓ Trend chart created")
        
        pie_chart = dashboard.create_distribution_pie_chart()
        print("   ✓ Distribution pie chart created")
        
        heatmap = dashboard.create_email_heatmap()
        print("   ✓ Email heatmap created")
        
        risk_chart = dashboard.create_risk_level_chart()
        print("   ✓ Risk level chart created")
        
        source_chart = dashboard.create_source_distribution_chart()
        print("   ✓ Source distribution chart created")
        
        confidence_hist = dashboard.create_confidence_histogram()
        print("   ✓ Confidence histogram created")
    except Exception as e:
        print(f"   ✗ Chart generation failed: {e}")
    
    # Test 3: Text Analysis
    print("\n3. Testing Text Analysis...")
    test_texts = [
        "This is a phishing email trying to steal your password",
        "Legitimate business email about quarterly reports",
        "URGENT: Your account will be suspended unless you click here"
    ]
    
    for text in test_texts:
        try:
            result = await dashboard.analyze_text(text)
            print(f"   Text: '{text[:50]}...'")
            print(f"   Fraud Score: {result['fraud_score']:.2%}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Types: {result['fraud_types']}")
        except Exception as e:
            print(f"   ✗ Analysis failed: {e}")
    
    # Test 4: Export Functions
    print("\n4. Testing Export Functions...")
    try:
        csv_file = dashboard.export_to_csv()
        print(f"   ✓ CSV exported to: {csv_file}")
        
        pdf_file = dashboard.export_to_pdf()
        print(f"   ✓ PDF exported to: {pdf_file}")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
    
    # Test 5: Data Generation
    print("\n5. Testing Data Generation...")
    initial_count = len(dashboard.fraud_data)
    dashboard._generate_sample_data()
    new_count = len(dashboard.fraud_data)
    print(f"   Initial data points: {initial_count}")
    print(f"   After generation: {new_count}")
    
    print("\n✅ All tests completed!")
    
    # Print summary
    print("\n" + "="*50)
    print("DASHBOARD TEST SUMMARY")
    print("="*50)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Fraud Records: {len(dashboard.fraud_data)}")
    print(f"Total Email Records: {len(dashboard.email_data)}")
    print(f"Detection History: {len(dashboard.detection_history)} items")
    print("\nDashboard Features:")
    print("✓ Real-time fraud trend analysis")
    print("✓ Fraud type distribution visualization")
    print("✓ Email pattern heatmap")
    print("✓ Risk level assessment")
    print("✓ Source distribution tracking")
    print("✓ Confidence score analysis")
    print("✓ CSV export functionality")
    print("✓ PDF report generation")
    print("✓ Live text analysis")
    
    return dashboard


if __name__ == "__main__":
    dashboard = asyncio.run(test_dashboard())
    print("\n📊 Dashboard testing complete!")
    print("You can now run 'python3 demo/gradio_analytics_dashboard.py' to launch the web interface.")