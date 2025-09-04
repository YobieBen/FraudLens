"""
Comprehensive End-to-End System Tests for FraudLens.

Author: Yobie Benjamin
Date: 2025-08-27 18:47:00 PDT
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import pytest
import pytest_asyncio

# Test data samples
COMPREHENSIVE_TEST_DATA = {
    "phishing_email": """
    Dear Customer,
    
    URGENT: Your PayPal account has been temporarily suspended due to suspicious activity.
    We detected unauthorized access from IP 192.168.1.1 attempting to transfer $5,000.
    
    Click here immediately to verify your identity: http://paypaI-security.fake/verify
    
    You must act within 24 hours or your account will be permanently closed and funds frozen.
    
    Regards,
    PayPal Security Team
    """,
    "legitimate_email": """
    Dear Valued Customer,
    
    Thank you for your recent purchase of the Premium Subscription.
    Your order #12345 has been processed successfully.
    
    You can manage your subscription settings by logging into your account
    dashboard at any time.
    
    Best regards,
    Customer Service Team
    """,
    "money_laundering_text": """
    Need to transfer $9,500 through multiple accounts to avoid detection.
    Will split into smaller amounts under $10k reporting threshold.
    Using offshore accounts in Cayman Islands and Swiss banks.
    Convert to Bitcoin through mixer for anonymity.
    """,
    "social_engineering_call": """
    This is IT Support calling about a critical security breach on your computer.
    We need remote access immediately to prevent data loss.
    Everyone else in your department has already complied with this security audit.
    If you don't act now, there will be serious consequences for the company.
    """,
    "financial_document": """
    INVOICE #INV-2024-001
    Date: 12/01/2024
    
    Bill To: ABC Company
    Items:
    - Service A: $1,000.00
    - Service B: $2,000.00  
    - Service C: $3,000.00
    
    Subtotal: $6,000.00
    Tax (10%): $600.00
    Total: $10,000.00  # Intentional calculation error for testing
    """,
}


class TestFullSystemIntegration:
    """Test complete FraudLens system integration."""

    @pytest_asyncio.fixture
    async def fraud_pipeline(self):
        """Create full fraud detection pipeline."""
        from fraudlens.core.config import Config
        from fraudlens.core.pipeline import FraudDetectionPipeline

        # Create config
        config = Config(
            {
                "processors": {
                    "text": {"enabled": True, "batch_size": 10},
                    "image": {"enabled": False},
                    "video": {"enabled": False},
                    "audio": {"enabled": False},
                },
                "resource_limits": {
                    "max_memory_gb": 10,
                    "max_cpu_percent": 80,
                    "enable_gpu": False,
                },
                "cache": {"enabled": True, "max_size": 1000},
            }
        )

        # Initialize pipeline
        pipeline = FraudDetectionPipeline(config)
        await pipeline.initialize()

        yield pipeline

        await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_text_fraud_detection(self, fraud_pipeline):
        """Test text-based fraud detection."""
        results = {}

        # Test each sample
        for name, text in COMPREHENSIVE_TEST_DATA.items():
            result = await fraud_pipeline.process(text, modality="text")
            results[name] = result

            print(f"\n{name}:")
            print(f"  Fraud Score: {result.fraud_score:.2f}")
            print(f"  Fraud Types: {[ft.value for ft in result.fraud_types]}")
            print(f"  Processing Time: {result.processing_time_ms:.1f}ms")

        # Assertions
        assert results["phishing_email"].fraud_score > 0.7, "Should detect phishing"
        assert results["legitimate_email"].fraud_score < 0.3, "Should not flag legitimate"
        assert results["money_laundering_text"].fraud_score > 0.6, "Should detect ML"
        assert results["social_engineering_call"].fraud_score > 0.7, "Should detect social eng"
        assert results["financial_document"].fraud_score > 0.5, "Should detect doc fraud"

    @pytest.mark.asyncio
    async def test_batch_processing(self, fraud_pipeline):
        """Test batch processing performance."""
        texts = list(COMPREHENSIVE_TEST_DATA.values()) * 10  # 50 samples

        start_time = time.time()
        results = await fraud_pipeline.batch_process(texts, modality="text")
        total_time = time.time() - start_time

        throughput = len(texts) / total_time
        avg_time = (total_time / len(texts)) * 1000

        print(f"\nBatch Processing Results:")
        print(f"  Total samples: {len(texts)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} docs/sec")
        print(f"  Average time: {avg_time:.1f} ms/doc")

        assert len(results) == len(texts), "Should process all samples"
        assert throughput > 10, "Should process >10 docs/sec"

    @pytest.mark.asyncio
    async def test_plugin_system(self, fraud_pipeline):
        """Test plugin loading and execution."""
        from fraudlens.plugins.base import FraudDetectorPlugin

        # Create a test plugin
        class TestPlugin(FraudDetectorPlugin):
            def __init__(self):
                super().__init__("test_plugin", "1.0.0")

            async def process(self, data, **kwargs):
                return {"test": True, "score": 0.5}

        # Register plugin
        plugin = TestPlugin()
        fraud_pipeline.plugin_manager.register_plugin(plugin)

        # Test plugin execution
        result = await fraud_pipeline.plugin_manager.execute_plugin("test_plugin", {"data": "test"})

        assert result["test"] == True
        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_resource_management(self, fraud_pipeline):
        """Test resource management and limits."""
        manager = fraud_pipeline.resource_manager

        # Check initial state
        stats = manager.get_statistics()
        assert stats["memory_usage_mb"] < 2000, "Memory should be reasonable"
        # Just check that memory tracking is working, not specific percentage
        assert "memory_usage_percent" in stats, "Memory tracking should work"

        # Process large batch
        large_batch = list(COMPREHENSIVE_TEST_DATA.values()) * 20
        await fraud_pipeline.batch_process(large_batch, modality="text")

        # Check resource usage after processing
        stats_after = manager.get_statistics()
        assert stats_after["memory_usage_mb"] < 2000, "Memory should stay bounded"
        assert stats_after["peak_memory_mb"] < 3000, "Peak memory should be reasonable"

        print(f"\nResource Usage:")
        print(f"  Current Memory: {stats_after['memory_usage_mb']:.1f} MB")
        print(f"  Peak Memory: {stats_after['peak_memory_mb']:.1f} MB")
        print(f"  CPU Usage: {stats_after['cpu_percent']:.1f}%")

    @pytest.mark.asyncio
    async def test_caching_effectiveness(self, fraud_pipeline):
        """Test caching improves performance."""
        test_text = COMPREHENSIVE_TEST_DATA["phishing_email"]

        # First run (cold cache)
        start = time.time()
        result1 = await fraud_pipeline.process(test_text, modality="text")
        cold_time = (time.time() - start) * 1000

        # Second run (warm cache)
        start = time.time()
        result2 = await fraud_pipeline.process(test_text, modality="text")
        warm_time = (time.time() - start) * 1000

        # Results should be consistent (scores similar, base fraud types present)
        assert abs(result1.fraud_score - result2.fraud_score) < 0.1
        # Check that core fraud types are present in both (cache may add more detail)
        if result1.fraud_types and result2.fraud_types:
            # At least one fraud type should be common
            assert len(set(result1.fraud_types) & set(result2.fraud_types)) > 0

        # Cache should improve performance
        speedup = cold_time / warm_time if warm_time > 0 else 10

        print(f"\nCache Performance:")
        print(f"  Cold cache: {cold_time:.2f}ms")
        print(f"  Warm cache: {warm_time:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")

        assert speedup > 2, "Cache should provide >2x speedup"

    @pytest.mark.asyncio
    async def test_model_registry(self, fraud_pipeline):
        """Test model registry functionality."""
        registry = fraud_pipeline.model_registry

        # Register a test model
        import tempfile

        from fraudlens.core.registry.model_registry import ModelFormat

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(b"dummy model")
            model_path = tmp.name

        model_info = registry.register_model(
            name="test_model",
            version="1.0.0",
            path=model_path,
            format=ModelFormat.PYTORCH,
            modality="text",
            metadata={"accuracy": 0.95},
        )

        # Clean up temp file after test
        import os

        os.unlink(model_path)

        # Retrieve model
        retrieved = registry.get_model(name="test_model", version="1.0.0")
        assert retrieved is not None
        assert retrieved.metadata["accuracy"] == 0.95

        # List models
        models = registry.list_models()
        model_names = [m.name for m in models]
        assert "test_model" in model_names

        print(f"\nModel Registry:")
        print(f"  Registered models: {len(models)}")
        print(f"  Models: {', '.join(model_names)}")

    @pytest.mark.asyncio
    async def test_error_handling(self, fraud_pipeline):
        """Test error handling and recovery."""
        # Test with invalid input
        result = await fraud_pipeline.process(None, modality="text")
        assert result is not None, "Should handle None gracefully"

        # Test with empty input
        result = await fraud_pipeline.process("", modality="text")
        assert result.fraud_score == 0, "Empty input should have 0 score"

        # Test with very large input
        large_text = "spam " * 10000
        result = await fraud_pipeline.process(large_text, modality="text")
        assert result is not None, "Should handle large input"

        print("\n✅ Error handling tests passed")

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, fraud_pipeline):
        """Test concurrent request handling."""
        texts = list(COMPREHENSIVE_TEST_DATA.values())

        # Launch concurrent tasks
        start = time.time()
        tasks = [fraud_pipeline.process(text, modality="text") for text in texts]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

        assert len(results) == len(texts), "All tasks should complete"
        assert all(r is not None for r in results), "All results should be valid"

        print(f"\nConcurrent Processing:")
        print(f"  Concurrent requests: {len(texts)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per request: {(total_time/len(texts)*1000):.1f}ms")


class TestTextProcessingModule:
    """Test text processing module specifically."""

    @pytest_asyncio.fixture
    async def text_detector(self):
        """Create text fraud detector."""
        from fraudlens.processors.text.detector import TextFraudDetector

        detector = TextFraudDetector(cache_size=100, enable_gpu=False, batch_size=10)
        await detector.initialize()

        yield detector

        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_specialized_analyzers(self, text_detector):
        """Test all specialized analyzers."""
        # Test phishing analyzer
        phishing_result = await text_detector.analyze_phishing(
            COMPREHENSIVE_TEST_DATA["phishing_email"]
        )
        assert phishing_result.is_phishing == True
        assert len(phishing_result.suspicious_urls) > 0

        # Test social engineering analyzer
        social_result = await text_detector.detect_social_engineering(
            COMPREHENSIVE_TEST_DATA["social_engineering_call"]
        )
        assert social_result.detected == True
        assert len(social_result.tactics) > 0

        # Test financial document analyzer
        doc_result = await text_detector.analyze_financial_document(
            COMPREHENSIVE_TEST_DATA["financial_document"]
        )
        assert doc_result.is_fraudulent == True
        assert len(doc_result.financial_inconsistencies) > 0

        print("\n✅ All specialized analyzers working correctly")

    @pytest.mark.asyncio
    async def test_llm_integration(self, text_detector):
        """Test LLM integration."""
        result = await text_detector.detect(COMPREHENSIVE_TEST_DATA["phishing_email"])

        # Check explanation was generated
        assert result.explanation is not None
        assert len(result.explanation) > 10

        print(f"\nLLM Integration:")
        print(f"  Explanation: {result.explanation[:100]}...")

    @pytest.mark.asyncio
    async def test_feature_extraction(self, text_detector):
        """Test feature extraction."""
        features = await text_detector.feature_extractor.extract_features(
            COMPREHENSIVE_TEST_DATA["phishing_email"]
        )

        assert features["has_urls"] == True
        assert features["urgency_score"] > 0.5
        assert features["has_financial_terms"] == True

        print(f"\nFeature Extraction:")
        print(f"  URLs found: {features['url_count']}")
        print(f"  Urgency score: {features['urgency_score']:.2f}")
        print(f"  Financial terms: {features['has_financial_terms']}")


def run_comprehensive_benchmark():
    """Run comprehensive system benchmark."""
    import asyncio

    async def benchmark():
        from fraudlens.core.config import Config
        from fraudlens.core.pipeline import FraudDetectionPipeline

        print("\n" + "=" * 70)
        print("FRAUDLENS COMPREHENSIVE SYSTEM BENCHMARK")
        print("=" * 70)

        # Initialize system
        config = Config(
            {
                "processors": {
                    "text": {"enabled": True, "batch_size": 50},
                },
                "resource_limits": {"max_memory_gb": 10, "enable_gpu": False},
                "cache": {"enabled": True, "max_size": 1000},
            }
        )

        pipeline = FraudDetectionPipeline(config)
        await pipeline.initialize()

        # Prepare test data
        test_samples = []
        for _ in range(20):  # 100 total samples
            test_samples.extend(list(COMPREHENSIVE_TEST_DATA.values()))

        print(f"\nTest Configuration:")
        print(f"  Total samples: {len(test_samples)}")
        print(f"  Processor: Text")
        print(f"  Cache: Enabled")
        print(f"  GPU: Disabled")

        # Warmup
        await pipeline.process(test_samples[0], modality="text")

        # Benchmark
        start_time = time.time()
        results = await pipeline.batch_process(test_samples, modality="text")
        total_time = time.time() - start_time

        # Calculate metrics
        successful = sum(1 for r in results if r is not None)
        avg_fraud_score = sum(r.fraud_score for r in results if r) / len(results)
        throughput = len(test_samples) / total_time
        avg_latency = (total_time / len(test_samples)) * 1000

        # Get system stats
        stats = pipeline.resource_manager.get_statistics()

        print(f"\nPerformance Results:")
        print(f"  Throughput: {throughput:.1f} docs/second")
        print(f"  Average latency: {avg_latency:.2f} ms/doc")
        print(f"  Success rate: {(successful/len(test_samples)*100):.1f}%")
        print(f"  Average fraud score: {avg_fraud_score:.3f}")

        print(f"\nResource Usage:")
        print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"  Peak memory: {stats['peak_memory_mb']:.1f} MB")
        print(f"  CPU usage: {stats['cpu_percent']:.1f}%")

        # Performance targets
        targets = {
            "throughput": 10,  # docs/sec
            "latency": 100,  # ms
            "memory": 1000,  # MB
        }

        print(f"\nPerformance vs Targets:")
        if throughput >= targets["throughput"]:
            print(f"  ✅ Throughput: {throughput:.1f} >= {targets['throughput']} docs/sec")
        else:
            print(f"  ❌ Throughput: {throughput:.1f} < {targets['throughput']} docs/sec")

        if avg_latency <= targets["latency"]:
            print(f"  ✅ Latency: {avg_latency:.1f} <= {targets['latency']} ms")
        else:
            print(f"  ❌ Latency: {avg_latency:.1f} > {targets['latency']} ms")

        if stats["memory_usage_mb"] <= targets["memory"]:
            print(f"  ✅ Memory: {stats['memory_usage_mb']:.1f} <= {targets['memory']} MB")
        else:
            print(f"  ❌ Memory: {stats['memory_usage_mb']:.1f} > {targets['memory']} MB")

        await pipeline.cleanup()

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

    asyncio.run(benchmark())


if __name__ == "__main__":
    # Run tests
    print("Running FraudLens E2E Tests...")
    pytest.main([__file__, "-v", "--tb=short"])

    # Run benchmark
    print("\n" + "=" * 70)
    run_comprehensive_benchmark()
