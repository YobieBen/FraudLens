#!/usr/bin/env python3
import asyncio
from fraudlens.core.pipeline import FraudDetectionPipeline

async def test():
    pipeline = FraudDetectionPipeline()
    await pipeline.initialize()
    
    test = "Your account has been compromised. Click here to secure it."
    result = await pipeline.process(test, modality="text")
    print(f"Fraud: {test}")
    print(f"  Score: {result.fraud_score:.1%}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Consistent: {'✅' if result.fraud_score > 0 else '❌'}")

asyncio.run(test())
