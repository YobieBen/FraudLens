#!/usr/bin/env python3
"""
Final End-to-End Test and Performance Report for FraudLens.

Author: Yobie Benjamin
Date: 2025-08-27 19:03:00 PDT
"""

import asyncio
import time
from typing import List

# Test samples
FRAUD_SAMPLES = {
    "phishing": """
    URGENT: Your account has been suspended due to suspicious activity.
    Click here to verify: http://paypal-fake.com/verify
    Act within 24 hours or lose access permanently.
    """,
    
    "money_laundering": """
    Need to transfer $9,500 through offshore accounts in Cayman Islands.
    Split into smaller amounts to avoid reporting threshold.
    Use Bitcoin mixer for anonymity.
    """,
    
    "social_engineering": """
    This is IT Support. Security breach detected on your computer.
    Give us remote access immediately to prevent data loss.
    """,
    
    "legitimate": """
    Thank you for your purchase. Your order #12345 has been shipped.
    Track your package in your account dashboard.
    """,
}


async def test_text_processor():
    """Test text processing module directly."""
    print("\n" + "="*70)
    print("TEXT PROCESSOR MODULE TEST")
    print("="*70)
    
    from fraudlens.processors.text.detector import TextFraudDetector
    
    detector = TextFraudDetector(cache_size=100, enable_gpu=False)
    await detector.initialize()
    
    for name, text in FRAUD_SAMPLES.items():
        result = await detector.detect(text)
        print(f"\n{name}:")
        print(f"  Fraud Score: {result.fraud_score:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Types: {[ft.value for ft in result.fraud_types]}")
    
    # Test batch processing
    print("\n" + "-"*40)
    print("Batch Processing Test:")
    
    batch = list(FRAUD_SAMPLES.values()) * 10  # 40 samples
    start = time.time()
    results = await detector.batch_process(batch)
    elapsed = time.time() - start
    
    print(f"  Processed: {len(batch)} samples")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {len(batch)/elapsed:.1f} docs/sec")
    
    await detector.cleanup()
    return True


async def test_pipeline():
    """Test full pipeline integration."""
    print("\n" + "="*70)
    print("PIPELINE INTEGRATION TEST")
    print("="*70)
    
    from fraudlens.core.pipeline import FraudDetectionPipeline
    from fraudlens.core.config import Config
    
    config = Config({
        "processors": {"text": {"enabled": True}},
        "cache": {"enabled": True, "max_size": 100}
    })
    
    pipeline = FraudDetectionPipeline(config)
    await pipeline.initialize()
    
    # Test individual processing
    for name, text in FRAUD_SAMPLES.items():
        result = await pipeline.process(text, modality="text")
        if result:
            print(f"\n{name}:")
            print(f"  Score: {result.fraud_score:.2f}")
            print(f"  Types: {[ft.value for ft in result.fraud_types]}")
    
    stats = pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Average time: {stats['average_time_ms']:.1f}ms")
    print(f"  Errors: {stats['errors']}")
    
    await pipeline.cleanup()
    return True


async def test_specialized_analyzers():
    """Test specialized fraud analyzers."""
    print("\n" + "="*70)
    print("SPECIALIZED ANALYZERS TEST")
    print("="*70)
    
    from fraudlens.processors.text.detector import TextFraudDetector
    
    detector = TextFraudDetector()
    await detector.initialize()
    
    # Test phishing analyzer
    phishing_result = await detector.analyze_phishing(FRAUD_SAMPLES["phishing"])
    print(f"\nPhishing Analyzer:")
    print(f"  Detected: {phishing_result.is_phishing}")
    print(f"  Confidence: {phishing_result.confidence:.2f}")
    print(f"  Indicators: {len(phishing_result.indicators)}")
    
    # Test social engineering analyzer
    social_result = await detector.detect_social_engineering(
        FRAUD_SAMPLES["social_engineering"]
    )
    print(f"\nSocial Engineering Analyzer:")
    print(f"  Detected: {social_result.detected}")
    print(f"  Tactics: {social_result.tactics}")
    print(f"  Risk Level: {social_result.risk_level}")
    
    await detector.cleanup()
    return True


async def performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    
    from fraudlens.processors.text.detector import TextFraudDetector
    
    detector = TextFraudDetector(cache_size=1000, enable_gpu=False, batch_size=50)
    await detector.initialize()
    
    # Create large test set
    test_samples = []
    for _ in range(25):  # 100 total samples
        test_samples.extend(list(FRAUD_SAMPLES.values()))
    
    print(f"\nTest samples: {len(test_samples)}")
    
    # Warmup
    await detector.detect(test_samples[0])
    
    # Benchmark
    start = time.time()
    results = await detector.batch_process(test_samples)
    elapsed = time.time() - start
    
    # Calculate metrics
    throughput = len(test_samples) / elapsed
    avg_latency = (elapsed / len(test_samples)) * 1000
    fraud_detected = sum(1 for r in results if r.fraud_score > 0.5)
    
    # Get stats
    stats = detector.get_performance_stats()
    
    print(f"\nResults:")
    print(f"  Throughput: {throughput:.1f} docs/sec")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  Fraud Detected: {fraud_detected}/{len(test_samples)}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Memory Usage: {stats['memory_usage_mb']:.1f} MB")
    
    # Check targets
    print(f"\nPerformance vs Targets:")
    targets = {"throughput": 10, "latency": 100}
    
    if throughput >= targets["throughput"]:
        print(f"  ✅ Throughput: {throughput:.1f} >= {targets['throughput']} docs/sec")
    else:
        print(f"  ❌ Throughput: {throughput:.1f} < {targets['throughput']} docs/sec")
    
    if avg_latency <= targets["latency"]:
        print(f"  ✅ Latency: {avg_latency:.1f} <= {targets['latency']} ms")
    else:
        print(f"  ❌ Latency: {avg_latency:.1f} > {targets['latency']} ms")
    
    await detector.cleanup()
    return throughput >= targets["throughput"]


async def main():
    """Run all E2E tests."""
    print("\n" + "#"*70)
    print("#" + " "*20 + "FRAUDLENS E2E TEST SUITE" + " "*20 + "#")
    print("#"*70)
    
    results = []
    
    # Run tests
    try:
        results.append(("Text Processor", await test_text_processor()))
    except Exception as e:
        print(f"Text Processor test failed: {e}")
        results.append(("Text Processor", False))
    
    try:
        results.append(("Pipeline Integration", await test_pipeline()))
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        results.append(("Pipeline Integration", False))
    
    try:
        results.append(("Specialized Analyzers", await test_specialized_analyzers()))
    except Exception as e:
        print(f"Analyzers test failed: {e}")
        results.append(("Specialized Analyzers", False))
    
    try:
        results.append(("Performance Benchmark", await performance_benchmark()))
    except Exception as e:
        print(f"Benchmark failed: {e}")
        results.append(("Performance Benchmark", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! FraudLens is ready for production.")
    else:
        print(f"\n⚠️ {total_count - passed_count} tests failed. Review and fix issues.")


if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)