#!/usr/bin/env python3
"""Test legitimate email detection."""

import asyncio
from fraudlens.core.pipeline import FraudDetectionPipeline

async def test_legitimate():
    pipeline = FraudDetectionPipeline()
    await pipeline.initialize()
    
    test_text = "This is a normal business email about the upcoming meeting."
    print(f"\nTesting: {test_text}")
    
    result = await pipeline.process(test_text, modality="text")
    print(f"\nPipeline result:")
    print(f"  fraud_score: {result.fraud_score}")
    print(f"  confidence: {result.confidence}")
    print(f"  fraud_types: {result.fraud_types}")
    print(f"  explanation: {result.explanation}")
    
    # Check consistency
    if result.fraud_score == 0 and "detected potential fraud" in result.explanation.lower():
        print("\n❌ INCONSISTENCY DETECTED!")
        print("  Score is 0 but explanation mentions fraud indicators")
    elif result.fraud_score > 0 and "no fraud" in result.explanation.lower():
        print("\n❌ INCONSISTENCY DETECTED!")
        print("  Score > 0 but explanation says no fraud")
    else:
        print("\n✅ Score and explanation are consistent")

if __name__ == "__main__":
    asyncio.run(test_legitimate())