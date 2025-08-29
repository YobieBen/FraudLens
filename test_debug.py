#!/usr/bin/env python3
"""Debug test to see what's happening with fraud detection."""

import asyncio
from fraudlens.core.pipeline import FraudDetectionPipeline

async def test_phishing():
    pipeline = FraudDetectionPipeline()
    await pipeline.initialize()
    
    test_text = "Your account has been compromised. Click here to secure it immediately."
    print(f"\nTesting: {test_text}")
    
    # Get the text detector directly
    text_detector = pipeline.processors.get("text")
    
    # Test the phishing analyzer directly
    phishing_result = await text_detector.analyze_phishing(test_text)
    print(f"\nDirect phishing analysis:")
    print(f"  is_phishing: {phishing_result.is_phishing}")
    print(f"  confidence: {phishing_result.confidence}")
    print(f"  indicators: {phishing_result.indicators}")
    
    # Now test through the full pipeline
    result = await pipeline.process(test_text, modality="text")
    print(f"\nFull pipeline result:")
    print(f"  fraud_score: {result.fraud_score}")
    print(f"  confidence: {result.confidence}")
    print(f"  fraud_types: {result.fraud_types}")
    print(f"  explanation: {result.explanation}")
    
    # Check the evidence
    if hasattr(result, 'evidence') and result.evidence:
        print(f"\nEvidence details:")
        if 'phishing_result' in result.evidence:
            pr = result.evidence['phishing_result']
            print(f"  Phishing result in evidence:")
            print(f"    is_phishing: {pr.get('is_phishing')}")
            print(f"    confidence: {pr.get('confidence')}")

if __name__ == "__main__":
    asyncio.run(test_phishing())