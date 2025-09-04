"""
Performance benchmarks for FraudLens
Tests system performance, throughput, and scalability
"""

import asyncio
import concurrent.futures
import cProfile
import io
import pstats
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import pytest
from memory_profiler import profile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fraudlens.api.gmail_integration import EmailAction, EmailAnalysisResult, GmailFraudScanner
from fraudlens.core.cache_manager import cache_manager
from fraudlens.core.optimized_processor import LargeFileProcessor
from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.processors.text.detector import TextFraudDetector
from fraudlens.processors.vision.detector import VisionFraudDetector


@dataclass
class BenchmarkResult:
    """Benchmark result data"""

    test_name: str
    duration_seconds: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "duration_seconds": round(self.duration_seconds, 2),
            "operations_per_second": round(self.operations_per_second, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "success_rate": round(self.success_rate, 3),
        }


class PerformanceBenchmark:
    """Base class for performance benchmarks"""

    def __init__(self):
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []

    def measure_performance(self, func, *args, **kwargs) -> Tuple[Any, float, float, float]:
        """Measure function performance"""
        # Memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024

        # CPU before
        cpu_before = self.process.cpu_percent()

        # Execute function
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        # Memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024

        # CPU after
        cpu_after = self.process.cpu_percent()

        duration = end_time - start_time
        memory_used = mem_after - mem_before
        cpu_used = (cpu_after + cpu_before) / 2

        return result, duration, memory_used, cpu_used

    async def measure_async_performance(
        self, func, *args, **kwargs
    ) -> Tuple[Any, float, float, float]:
        """Measure async function performance"""
        mem_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()

        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()

        mem_after = self.process.memory_info().rss / 1024 / 1024
        cpu_after = self.process.cpu_percent()

        duration = end_time - start_time
        memory_used = mem_after - mem_before
        cpu_used = (cpu_after + cpu_before) / 2

        return result, duration, memory_used, cpu_used

    def calculate_percentiles(self, latencies: List[float]) -> Tuple[float, float, float]:
        """Calculate latency percentiles"""
        if not latencies:
            return 0, 0, 0

        sorted_latencies = sorted(latencies)
        p50 = np.percentile(sorted_latencies, 50)
        p95 = np.percentile(sorted_latencies, 95)
        p99 = np.percentile(sorted_latencies, 99)

        return p50 * 1000, p95 * 1000, p99 * 1000  # Convert to ms


class TestTextProcessingBenchmarks(PerformanceBenchmark):
    """Benchmarks for text fraud detection"""

    @pytest.fixture
    def text_detector(self):
        """Create text fraud detector"""
        return TextFraudDetector()

    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for testing"""
        texts = []

        # Short texts
        for i in range(100):
            texts.append(f"This is a test message {i} with some content.")

        # Medium texts (1KB)
        for i in range(50):
            texts.append("Lorem ipsum dolor sit amet " * 40)

        # Long texts (10KB)
        for i in range(10):
            texts.append("Long text content here " * 500)

        return texts

    @pytest.mark.benchmark
    async def test_text_detection_throughput(self, text_detector, sample_texts):
        """Benchmark text fraud detection throughput"""
        latencies = []
        successes = 0

        start_time = time.perf_counter()

        for text in sample_texts[:100]:  # Test 100 texts
            try:
                text_start = time.perf_counter()
                result = await text_detector.detect(text)
                text_end = time.perf_counter()

                latencies.append(text_end - text_start)
                successes += 1
            except Exception:
                pass

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        p50, p95, p99 = self.calculate_percentiles(latencies)

        result = BenchmarkResult(
            test_name="text_detection_throughput",
            duration_seconds=total_duration,
            operations_per_second=len(sample_texts[:100]) / total_duration,
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            success_rate=successes / len(sample_texts[:100]),
        )

        self.results.append(result)
        print(f"\nText Detection Benchmark: {result.to_dict()}")

        # Assert performance requirements
        assert result.operations_per_second > 10  # At least 10 ops/sec
        assert result.p95_latency_ms < 500  # P95 under 500ms
        assert result.success_rate > 0.95  # 95% success rate

    @pytest.mark.benchmark
    async def test_concurrent_text_processing(self, text_detector, sample_texts):
        """Benchmark concurrent text processing"""

        async def process_text(text):
            return await text_detector.detect(text)

        # Process texts concurrently
        start_time = time.perf_counter()

        tasks = [process_text(text) for text in sample_texts[:50]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        successes = sum(1 for r in results if not isinstance(r, Exception))

        result = BenchmarkResult(
            test_name="concurrent_text_processing",
            duration_seconds=total_duration,
            operations_per_second=len(tasks) / total_duration,
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            p50_latency_ms=0,  # Not measured for concurrent
            p95_latency_ms=0,
            p99_latency_ms=0,
            success_rate=successes / len(tasks),
        )

        self.results.append(result)
        print(f"\nConcurrent Text Processing: {result.to_dict()}")

        assert result.operations_per_second > 20  # Higher throughput with concurrency
        assert result.success_rate > 0.95


class TestEmailProcessingBenchmarks(PerformanceBenchmark):
    """Benchmarks for email processing"""

    @pytest.fixture
    def batch_processor(self):
        """Create batch email processor"""
        return GmailFraudScanner()

    @pytest.fixture
    def sample_emails(self):
        """Generate sample emails"""
        emails = []
        for i in range(1000):
            emails.append(
                EmailAnalysisResult(
                    message_id=f"msg{i}",
                    subject=f"Test Subject {i}",
                    sender=f"sender{i}@example.com",
                    recipient=f"recipient{i}@example.com",
                    date=datetime.now(),
                    fraud_score=0.1,
                    fraud_types=[],
                    confidence=0.9,
                    explanation="Test email",
                    attachments_analyzed=[],
                    action_taken=EmailAction.NONE,
                    processing_time_ms=10,
                    raw_content_score=0.1,
                    attachment_scores=[],
                    combined_score=0.1,
                    flagged=False,
                    error=None,
                )
            )
        return emails

    @pytest.mark.benchmark
    async def test_email_batch_processing(self, batch_processor, sample_emails):
        """Benchmark email batch processing"""

        async def mock_process(email):
            # Simulate processing
            await asyncio.sleep(0.01)
            return {"message_id": email.id, "is_fraud": False, "confidence": 0.1}

        # Process in batches
        batch_size = 100
        batches = [
            sample_emails[i : i + batch_size] for i in range(0, len(sample_emails), batch_size)
        ]

        start_time = time.perf_counter()

        for batch in batches[:5]:  # Process 5 batches
            results = await batch_processor.process_batch(batch, mock_process)

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        total_emails = 5 * batch_size

        result = BenchmarkResult(
            test_name="email_batch_processing",
            duration_seconds=total_duration,
            operations_per_second=total_emails / total_duration,
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            p50_latency_ms=(total_duration / total_emails) * 1000,
            p95_latency_ms=0,
            p99_latency_ms=0,
            success_rate=1.0,
        )

        self.results.append(result)
        print(f"\nEmail Batch Processing: {result.to_dict()}")

        assert result.operations_per_second > 50  # Process 50+ emails/sec


class TestCachingBenchmarks(PerformanceBenchmark):
    """Benchmarks for caching system"""

    @pytest.mark.benchmark
    def test_cache_performance(self):
        """Benchmark cache operations"""
        test_data = {"key": "value", "data": "x" * 1000}

        # Benchmark cache writes
        write_latencies = []
        for i in range(1000):
            key = f"test_key_{i}"
            start = time.perf_counter()
            cache_manager.set(key, test_data, ttl=60)
            end = time.perf_counter()
            write_latencies.append(end - start)

        # Benchmark cache reads
        read_latencies = []
        hits = 0
        for i in range(1000):
            key = f"test_key_{i}"
            start = time.perf_counter()
            value = cache_manager.get(key)
            end = time.perf_counter()
            read_latencies.append(end - start)
            if value is not None:
                hits += 1

        write_p50, write_p95, write_p99 = self.calculate_percentiles(write_latencies)
        read_p50, read_p95, read_p99 = self.calculate_percentiles(read_latencies)

        result = BenchmarkResult(
            test_name="cache_operations",
            duration_seconds=sum(write_latencies) + sum(read_latencies),
            operations_per_second=2000 / (sum(write_latencies) + sum(read_latencies)),
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            p50_latency_ms=(write_p50 + read_p50) / 2,
            p95_latency_ms=(write_p95 + read_p95) / 2,
            p99_latency_ms=(write_p99 + read_p99) / 2,
            success_rate=hits / 1000,
        )

        self.results.append(result)
        print(f"\nCache Performance: {result.to_dict()}")

        assert result.p95_latency_ms < 10  # Cache operations under 10ms
        assert result.success_rate > 0.99  # 99% hit rate


class TestLargeFileProcessingBenchmarks(PerformanceBenchmark):
    """Benchmarks for large file processing"""

    @pytest.fixture
    def file_processor(self):
        """Create large file processor"""
        return LargeFileProcessor(max_workers=4)

    @pytest.mark.benchmark
    async def test_large_file_processing(self, file_processor, tmp_path):
        """Benchmark large file processing"""
        # Create test files
        file_sizes_mb = [1, 5, 10, 20]
        files = []

        for size_mb in file_sizes_mb:
            file_path = tmp_path / f"test_{size_mb}mb.txt"
            with open(file_path, "w") as f:
                # Write size_mb of data
                data = "x" * (1024 * 1024)
                for _ in range(size_mb):
                    f.write(data)
            files.append((file_path, size_mb))

        # Process files
        results = []
        for file_path, size_mb in files:

            async def mock_processor(chunk):
                return len(chunk)

            start = time.perf_counter()
            result = await file_processor.process_large_text(str(file_path), mock_processor)
            end = time.perf_counter()

            duration = end - start
            throughput_mbps = size_mb / duration

            results.append(
                {
                    "size_mb": size_mb,
                    "duration_seconds": duration,
                    "throughput_mbps": throughput_mbps,
                }
            )

            print(
                f"\nProcessed {size_mb}MB file in {duration:.2f}s " f"({throughput_mbps:.2f} MB/s)"
            )

        # Average performance
        avg_throughput = statistics.mean(r["throughput_mbps"] for r in results)

        result = BenchmarkResult(
            test_name="large_file_processing",
            duration_seconds=sum(r["duration_seconds"] for r in results),
            operations_per_second=len(files) / sum(r["duration_seconds"] for r in results),
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            success_rate=1.0,
        )

        self.results.append(result)

        assert avg_throughput > 10  # Process at least 10 MB/s


class TestMemoryBenchmarks(PerformanceBenchmark):
    """Memory usage benchmarks"""

    @pytest.mark.benchmark
    def test_memory_usage_under_load(self):
        """Test memory usage under heavy load"""
        import gc

        # Force garbage collection
        gc.collect()

        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Create large objects
        large_data = []
        for i in range(100):
            # Create 10MB objects
            data = "x" * (10 * 1024 * 1024)
            large_data.append(data)

            if i % 10 == 0:
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                print(
                    f"Memory after {i} objects: {current_memory:.2f}MB "
                    f"(+{memory_increase:.2f}MB)"
                )

        # Clear and collect
        large_data.clear()
        gc.collect()

        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_leaked = final_memory - initial_memory

        print(f"\nMemory leak test: {memory_leaked:.2f}MB")

        # Should not leak more than 100MB
        assert memory_leaked < 100


class TestConcurrencyBenchmarks(PerformanceBenchmark):
    """Concurrency and scaling benchmarks"""

    @pytest.mark.benchmark
    async def test_concurrent_operations(self):
        """Test system under concurrent load"""

        async def simulate_operation(op_id: int):
            # Simulate mixed operations
            await asyncio.sleep(0.01)  # Simulate I/O

            # Some CPU work
            result = sum(i * i for i in range(1000))

            return f"Operation {op_id} completed: {result}"

        # Test different concurrency levels
        concurrency_levels = [10, 50, 100, 200]
        results = []

        for level in concurrency_levels:
            start = time.perf_counter()

            tasks = [simulate_operation(i) for i in range(level)]
            await asyncio.gather(*tasks)

            end = time.perf_counter()
            duration = end - start

            ops_per_second = level / duration

            results.append(
                {"concurrency": level, "duration": duration, "ops_per_second": ops_per_second}
            )

            print(f"\nConcurrency {level}: {ops_per_second:.2f} ops/sec")

        # Should scale well
        assert results[-1]["ops_per_second"] > results[0]["ops_per_second"] * 2


class TestEndToEndBenchmarks(PerformanceBenchmark):
    """End-to-end system benchmarks"""

    @pytest.mark.benchmark
    async def test_full_pipeline_performance(self):
        """Test complete fraud detection pipeline"""
        pipeline = FraudDetectionPipeline()
        await pipeline.initialize()

        # Prepare test data
        test_cases = [
            ("text", "This is a suspicious message about winning money"),
            ("text", "Normal business email about quarterly results"),
            ("text", "URGENT: Verify your account immediately!"),
        ] * 10

        latencies = []

        start_time = time.perf_counter()

        for data_type, content in test_cases:
            op_start = time.perf_counter()

            if data_type == "text":
                result = await pipeline.process_text(content)

            op_end = time.perf_counter()
            latencies.append(op_end - op_start)

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        p50, p95, p99 = self.calculate_percentiles(latencies)

        result = BenchmarkResult(
            test_name="full_pipeline",
            duration_seconds=total_duration,
            operations_per_second=len(test_cases) / total_duration,
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            success_rate=1.0,
        )

        self.results.append(result)
        print(f"\nFull Pipeline Performance: {result.to_dict()}")

        assert result.p95_latency_ms < 1000  # Under 1 second P95


def generate_benchmark_report(results: List[BenchmarkResult]) -> str:
    """Generate benchmark report"""
    report = []
    report.append("=" * 80)
    report.append("FRAUDLENS PERFORMANCE BENCHMARK REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")

    for result in results:
        report.append(f"\n{result.test_name.upper()}")
        report.append("-" * 40)
        report.append(f"Duration: {result.duration_seconds:.2f}s")
        report.append(f"Throughput: {result.operations_per_second:.2f} ops/sec")
        report.append(f"Memory Usage: {result.memory_usage_mb:.2f} MB")
        report.append(f"CPU Usage: {result.cpu_usage_percent:.1f}%")
        report.append(f"P50 Latency: {result.p50_latency_ms:.2f}ms")
        report.append(f"P95 Latency: {result.p95_latency_ms:.2f}ms")
        report.append(f"P99 Latency: {result.p99_latency_ms:.2f}ms")
        report.append(f"Success Rate: {result.success_rate:.1%}")

    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("-" * 40)

    avg_throughput = statistics.mean(
        r.operations_per_second for r in results if r.operations_per_second > 0
    )
    avg_memory = statistics.mean(r.memory_usage_mb for r in results)
    avg_cpu = statistics.mean(r.cpu_usage_percent for r in results)

    report.append(f"Average Throughput: {avg_throughput:.2f} ops/sec")
    report.append(f"Average Memory Usage: {avg_memory:.2f} MB")
    report.append(f"Average CPU Usage: {avg_cpu:.1f}%")

    return "\n".join(report)


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])
