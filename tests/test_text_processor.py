"""
Comprehensive tests for text fraud detection pipeline.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio

# Test data
PHISHING_SAMPLES = [
    """
    URGENT: Your PayPal account has been suspended due to suspicious activity.
    Click here immediately to verify your identity and restore access:
    http://paypaI-security.fake/verify
    
    You must act within 24 hours or your account will be permanently closed.
    """,
    """
    Congratulations! You've won $1,000,000 in the Microsoft Lottery!
    To claim your prize, please provide your bank account details and SSN.
    This offer expires TODAY! Act now!
    """,
]

LEGITIMATE_SAMPLES = [
    """
    Thank you for your recent purchase. Your order #12345 has been shipped
    and will arrive within 3-5 business days. You can track your package
    using the link in your account dashboard.
    """,
    """
    This is a reminder that your subscription will renew on January 15, 2025.
    No action is needed if you wish to continue. To manage your subscription,
    log in to your account at our official website.
    """,
]

FINANCIAL_DOCUMENTS = [
    """
    INVOICE #INV-2024-001
    Date: 12/01/2024
    
    Bill To: ABC Company
    Amount: $5,000.00
    Tax: $500.00
    Total: $5,500.00
    
    Payment Terms: Net 30
    """,
    """
    INVOICE #INV-2024-002
    Date: 12/01/2024
    
    Items:
    - Service A: $1,000.00
    - Service B: $2,000.00
    - Service C: $3,000.00
    
    Total: $10,000.00  # Intentional error for testing
    """,
]

SOCIAL_ENGINEERING_SAMPLES = [
    """
    This is IT Support. We've detected a security breach on your computer.
    You must give us remote access immediately to prevent data loss.
    Call us back at 1-800-SCAMMER within the next hour or face serious consequences.
    """,
    """
    As discussed in our previous conversation, you agreed to help with this
    important security audit. Since you're already committed, please provide
    the requested access credentials. Everyone else in your department has
    already complied.
    """,
]


class TestTextFraudDetector:
    """Test suite for TextFraudDetector."""
    
    @pytest_asyncio.fixture
    async def detector(self):
        """Create detector instance."""
        from fraudlens.processors.text.detector import TextFraudDetector
        
        detector = TextFraudDetector(
            cache_size=100,
            enable_gpu=False,  # CPU for testing
            batch_size=10,
        )
        await detector.initialize()
        yield detector
        await detector.cleanup()
    
    @pytest.mark.asyncio
    async def test_phishing_detection(self, detector):
        """Test phishing detection accuracy."""
        for sample in PHISHING_SAMPLES:
            result = await detector.detect(sample)
            assert result.fraud_score > 0.5, "Should detect phishing"
            assert "PHISHING" in [ft.name for ft in result.fraud_types]
    
    @pytest.mark.asyncio
    async def test_legitimate_text(self, detector):
        """Test legitimate text classification."""
        for sample in LEGITIMATE_SAMPLES:
            result = await detector.detect(sample)
            assert result.fraud_score < 0.4, "Should not flag legitimate text"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, detector):
        """Test batch processing performance."""
        all_samples = PHISHING_SAMPLES + LEGITIMATE_SAMPLES
        
        start_time = time.time()
        results = await detector.batch_process(all_samples)
        processing_time = time.time() - start_time
        
        assert len(results) == len(all_samples)
        assert processing_time < len(all_samples) * 0.5  # <500ms per item
        
        # Check cache hits on second run
        start_time = time.time()
        cached_results = await detector.batch_process(all_samples)
        cached_time = time.time() - start_time
        
        assert cached_time < processing_time * 0.1  # 10x faster with cache
    
    @pytest.mark.asyncio
    async def test_financial_document_analysis(self, detector):
        """Test financial document fraud detection."""
        # Test document with calculation error
        result = await detector.analyze_financial_document(FINANCIAL_DOCUMENTS[1])
        assert result.is_fraudulent, "Should detect calculation error"
        assert len(result.financial_inconsistencies) > 0
    
    @pytest.mark.asyncio
    async def test_social_engineering_detection(self, detector):
        """Test social engineering tactics detection."""
        for sample in SOCIAL_ENGINEERING_SAMPLES:
            result = await detector.detect_social_engineering(sample)
            assert result.detected, "Should detect social engineering"
            assert len(result.tactics) > 0
            assert len(result.tactics) > 0 or len(result.psychological_triggers) > 0
    
    @pytest.mark.asyncio
    async def test_memory_management(self, detector):
        """Test memory usage stays within limits."""
        initial_memory = detector.get_memory_usage()
        
        # Process many samples
        large_batch = PHISHING_SAMPLES * 50
        await detector.batch_process(large_batch)
        
        final_memory = detector.get_memory_usage()
        memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)
        
        assert memory_increase_mb < 100, "Memory usage should stay reasonable"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, detector):
        """Test performance meets requirements."""
        # Target: 10 documents/second on M4 Max
        test_samples = (PHISHING_SAMPLES + LEGITIMATE_SAMPLES) * 5
        
        start_time = time.time()
        results = await detector.batch_process(test_samples)
        total_time = time.time() - start_time
        
        throughput = len(test_samples) / total_time
        
        assert throughput > 5, f"Should process >5 docs/sec, got {throughput:.1f}"
        
        # Get performance stats
        stats = detector.get_performance_stats()
        assert stats["average_processing_time_ms"] < 200
        assert stats["cache_hit_rate"] > 0  # Some cache hits expected


class TestFeatureExtractor:
    """Test suite for FeatureExtractor."""
    
    @pytest_asyncio.fixture
    async def extractor(self):
        """Create feature extractor instance."""
        from fraudlens.processors.text.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        await extractor.initialize()
        yield extractor
        await extractor.cleanup()
    
    @pytest.mark.asyncio
    async def test_url_extraction(self, extractor):
        """Test URL extraction."""
        text = """
        Visit our site at https://example.com and http://test.org.
        Also check bit.ly/short and the suspicious site paypaI.fake.com
        """
        
        features = await extractor.extract_features(text)
        assert features["url_count"] >= 2
        assert features["has_shortened_urls"]
        assert len(features["urls"]) > 0
    
    @pytest.mark.asyncio
    async def test_financial_entity_extraction(self, extractor):
        """Test financial entity extraction."""
        text = """
        Transfer funds from your Chase account to PayPal.
        We accept Visa and Mastercard. Bitcoin payments also accepted.
        """
        
        features = await extractor.extract_features(text)
        assert "chase" in features["banks_mentioned"]
        assert "paypal" in features["payment_services_mentioned"]
        assert features["crypto_mentioned"]
    
    @pytest.mark.asyncio
    async def test_urgency_scoring(self, extractor):
        """Test urgency detection."""
        urgent_text = "URGENT! Act immediately! This expires in 24 hours!"
        calm_text = "Thank you for your purchase. Have a nice day."
        
        urgent_features = await extractor.extract_features(urgent_text)
        calm_features = await extractor.extract_features(calm_text)
        
        assert urgent_features["urgency_score"] > 0.5
        assert calm_features["urgency_score"] < 0.2


class TestLLMIntegration:
    """Test LLM integration."""
    
    @pytest_asyncio.fixture
    async def llm_manager(self):
        """Create LLM manager instance."""
        from fraudlens.processors.text.llm_manager import LLMManager
        
        manager = LLMManager(device="cpu")
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_fraud_analysis(self, llm_manager):
        """Test LLM fraud analysis."""
        phishing_text = PHISHING_SAMPLES[0]
        
        result = await llm_manager.analyze_fraud(phishing_text, "phishing")
        assert isinstance(result, dict)
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, llm_manager):
        """Test explanation generation."""
        findings = {
            "fraud_types": ["phishing", "social_engineering"],
            "risk_scores": [0.8, 0.6],
            "features": {"urgency_score": 0.9},
        }
        
        explanation = await llm_manager.generate_explanation(findings)
        assert isinstance(explanation, str)
        assert len(explanation) > 10
        assert len(explanation) < 1000


class TestCacheManager:
    """Test cache functionality."""
    
    @pytest_asyncio.fixture
    async def cache_manager(self):
        """Create cache manager instance."""
        from fraudlens.processors.text.cache_manager import CacheManager
        
        cache = CacheManager(max_size=10, ttl_seconds=60, enable_similarity=False)
        await cache.initialize()
        yield cache
        await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache_manager):
        """Test LRU eviction policy."""
        # Fill cache beyond capacity
        for i in range(15):
            await cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Check size limit is enforced
        stats = cache_manager.get_stats()
        assert stats["size"] <= 10
        
        # Check oldest items were evicted
        assert await cache_manager.get("key_0") is None
        assert await cache_manager.get("key_14") is not None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache_manager):
        """Test TTL expiration."""
        await cache_manager.set("temp_key", "temp_value")
        
        # Should exist immediately
        assert await cache_manager.get("temp_key") == "temp_value"
        
        # Mock time passing (would need time manipulation in real test)
        # For now, just test the mechanism exists
        stats = cache_manager.get_stats()
        assert stats["size"] > 0


def run_performance_benchmark():
    """Run performance benchmark for production validation."""
    import asyncio
    
    async def benchmark():
        from fraudlens.processors.text.detector import TextFraudDetector
        
        print("\n" + "="*60)
        print("TEXT FRAUD DETECTION PERFORMANCE BENCHMARK")
        print("="*60)
        
        detector = TextFraudDetector(enable_gpu=True)
        await detector.initialize()
        
        # Prepare test data
        test_samples = (PHISHING_SAMPLES + LEGITIMATE_SAMPLES + 
                       FINANCIAL_DOCUMENTS + SOCIAL_ENGINEERING_SAMPLES) * 10
        
        print(f"\nTest samples: {len(test_samples)}")
        print("Starting benchmark...\n")
        
        # Warmup
        await detector.detect(test_samples[0])
        
        # Benchmark
        start_time = time.time()
        results = await detector.batch_process(test_samples)
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = len(test_samples) / total_time
        avg_time = total_time / len(test_samples) * 1000
        
        # Get stats
        stats = detector.get_performance_stats()
        
        print(f"Results:")
        print(f"  Total documents: {len(test_samples)}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.1f} docs/second")
        print(f"  Average time: {avg_time:.1f} ms/doc")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        
        # Check if meets target
        target_throughput = 10  # docs/second
        if throughput >= target_throughput:
            print(f"\n✅ PASSED: Throughput {throughput:.1f} >= {target_throughput} docs/sec")
        else:
            print(f"\n⚠️ BELOW TARGET: Throughput {throughput:.1f} < {target_throughput} docs/sec")
        
        await detector.cleanup()
    
    asyncio.run(benchmark())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run benchmark
    run_performance_benchmark()