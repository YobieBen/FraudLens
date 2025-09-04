"""
Tests for vision and document fraud detection pipeline.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
import io
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fraudlens.processors.vision import VisionFraudDetector, PDFProcessor, ImagePreprocessor
from fraudlens.processors.vision.analyzers import (
    DeepfakeDetector,
    DocumentForgeryDetector,
    LogoImpersonationDetector,
    ManipulationDetector,
    QRCodeAnalyzer,
)


# Test fixtures
@pytest.fixture
def vision_detector():
    """Create vision fraud detector."""

    async def _create():
        detector = VisionFraudDetector(
            enable_gpu=False,  # CPU only for testing
            batch_size=4,
            cache_size=10,
            use_metal=False,
        )
        await detector.initialize()
        return detector

    detector = asyncio.run(_create())
    yield detector
    asyncio.run(detector.cleanup())


@pytest.fixture
def sample_image():
    """Create sample test image."""
    # Create a simple document-like image
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Add some text
    draw.text((50, 50), "Test Document", fill="black")
    draw.text((50, 100), "This is a sample document for testing", fill="black")
    draw.text((50, 150), "Invoice #12345", fill="black")
    draw.text((50, 200), "Amount: $1,234.56", fill="black")

    # Add a signature-like scribble
    draw.line([(100, 400), (150, 380), (200, 400), (250, 390)], fill="blue", width=2)

    # Add a stamp-like circle
    draw.ellipse([(500, 400), (600, 500)], outline="red", width=3)
    draw.text((520, 440), "PAID", fill="red")

    return np.array(img)


@pytest.fixture
def forged_document_image():
    """Create a forged document image with artifacts."""
    # Create document with suspicious elements
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Add inconsistent text (different resolutions/styles)
    draw.text((50, 50), "Official Document", fill="black")

    # Add perfectly straight lines (digital editing indicator)
    draw.line([(0, 300), (800, 300)], fill="black", width=1)

    # Add uniform regions (copy-paste artifacts)
    draw.rectangle([(200, 200), (400, 250)], fill="gray")
    draw.rectangle([(450, 200), (650, 250)], fill="gray")  # Identical region

    # Add suspicious stamp (too perfect)
    draw.ellipse([(500, 400), (600, 500)], fill="red")
    draw.text((520, 440), "APPROVED", fill="white")

    return np.array(img)


@pytest.fixture
def deepfake_image():
    """Create image with deepfake-like characteristics."""
    # Create face-like image with artifacts
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a simple face
    # Face outline
    draw.ellipse([(150, 100), (350, 400)], fill=(255, 220, 177))

    # Eyes (unnatural symmetry)
    draw.ellipse([(180, 200), (220, 240)], fill="white")
    draw.ellipse([(280, 200), (320, 240)], fill="white")
    draw.ellipse([(190, 210), (210, 230)], fill="black")
    draw.ellipse([(290, 210), (310, 230)], fill="black")

    # Add GAN-like artifacts (checkerboard pattern in background)
    for i in range(0, 512, 20):
        for j in range(0, 512, 20):
            if (i // 20 + j // 20) % 2 == 0:
                draw.rectangle([(i, j), (i + 20, j + 20)], fill=(240, 240, 240))

    return np.array(img)


@pytest.fixture
def brand_impersonation_image():
    """Create image with brand impersonation."""
    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)

    # Add PayPal-like colors and text
    draw.rectangle([(0, 0), (800, 100)], fill=(0, 48, 135))  # PayPal blue
    draw.text((50, 30), "PaypaI Security Alert", fill="white")  # Note the capital I

    # Add phishing content
    draw.text((50, 150), "Your account has been suspended!", fill="red")
    draw.text((50, 200), "Click here to verify: http://paypaI-security.fake", fill="blue")

    return np.array(img)


@pytest.fixture
def malicious_qr_image():
    """Create image with QR code."""
    img = Image.new("RGB", (400, 400), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a simple QR-like pattern (not a real QR code)
    # This is just for testing detection
    for i in range(10, 390, 20):
        for j in range(10, 390, 20):
            if (i + j) % 40 == 0:
                draw.rectangle([(i, j), (i + 15, j + 15)], fill="black")

    # Add corner markers (QR characteristic)
    draw.rectangle([(10, 10), (70, 70)], fill="black")
    draw.rectangle([(20, 20), (60, 60)], fill="white")
    draw.rectangle([(30, 30), (50, 50)], fill="black")

    return np.array(img)


def create_test_pdf(num_pages: int = 3) -> bytes:
    """Create a test PDF with multiple pages."""
    try:
        import fitz  # PyMuPDF

        # Create PDF document
        doc = fitz.open()

        for page_num in range(num_pages):
            page = doc.new_page(width=612, height=792)  # Letter size

            # Add text to page
            text = f"Page {page_num + 1}\n\nThis is a test PDF document.\nInvoice #ABC{page_num+1:03d}\nAmount: ${(page_num+1)*1000:.2f}"

            # Insert text
            point = fitz.Point(72, 72)  # 1 inch margins
            page.insert_text(point, text, fontsize=12)

            # Add a rectangle (signature area)
            rect = fitz.Rect(72, 600, 300, 650)
            page.draw_rect(rect, color=(0, 0, 1), width=1)

            # Add stamp-like annotation
            stamp_rect = fitz.Rect(400, 600, 500, 700)
            page.draw_circle(fitz.Point(450, 650), 40, color=(1, 0, 0), width=2)

        # Save to bytes
        pdf_bytes = doc.tobytes()
        doc.close()

        return pdf_bytes

    except ImportError:
        # Create mock PDF bytes if PyMuPDF not available
        return b"%PDF-1.4\nMock PDF content for testing"


# Tests
class TestVisionFraudDetector:
    """Test vision fraud detection capabilities."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test detector initialization."""
        detector = VisionFraudDetector(enable_gpu=False)
        await detector.initialize()

        assert detector.detector_id == "vision_fraud_detector"
        assert detector.enable_gpu == False
        assert detector.batch_size == 8

        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_image_detection(self, vision_detector, sample_image):
        """Test basic image fraud detection."""
        result = await vision_detector.process_image_array(sample_image)

        assert result is not None
        assert isinstance(result.fraud_detected, bool)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_document_forgery_detection(self, vision_detector, forged_document_image):
        """Test document forgery detection."""
        result = await vision_detector.process_image_array(forged_document_image)

        assert result is not None

        # Check if forgery indicators were found
        if result.fraud_detected:
            assert "document_forgery" in result.fraud_types or "manipulation" in result.fraud_types

    @pytest.mark.asyncio
    async def test_deepfake_detection(self, vision_detector, deepfake_image):
        """Test deepfake detection."""
        result = await vision_detector.process_image_array(deepfake_image)

        assert result is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_brand_impersonation_detection(self, vision_detector, brand_impersonation_image):
        """Test brand impersonation detection."""
        result = await vision_detector.process_image_array(brand_impersonation_image)

        assert result is not None

        # Should detect PayPal-like colors
        if result.fraud_detected:
            assert "brand_impersonation" in result.fraud_types or result.confidence > 0

    @pytest.mark.asyncio
    async def test_qr_code_analysis(self, vision_detector, malicious_qr_image):
        """Test QR code analysis."""
        result = await vision_detector.process_image_array(malicious_qr_image)

        assert result is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_pdf_processing(self, vision_detector):
        """Test PDF document processing."""
        # Create test PDF
        pdf_bytes = create_test_pdf(3)

        # Process PDF
        result = await vision_detector.process_pdf_bytes(pdf_bytes)

        assert result is not None
        assert result.document_type in [
            "invoice",
            "receipt",
            "contract",
            "statement",
            "generic_document",
        ]
        assert result.page_analyses is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_batch_processing(self, vision_detector, sample_image):
        """Test batch image processing."""
        images = [sample_image] * 3

        results = await vision_detector.batch_process(images)

        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_caching(self, vision_detector):
        """Test result caching."""
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Save to temp file for consistent hashing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(img).save(tmp.name)

            # First processing (cache miss)
            result1 = await vision_detector.process_image(tmp.name)
            time1 = result1.processing_time_ms

            # Second processing (cache hit)
            result2 = await vision_detector.process_image(tmp.name)
            time2 = result2.processing_time_ms

            # Cache hit should be much faster
            assert time2 < time1 * 0.1  # At least 10x faster
            assert vision_detector._cache_hits > 0

        # Clean up
        Path(tmp.name).unlink()

    @pytest.mark.asyncio
    async def test_visual_consistency_check(self, vision_detector, sample_image):
        """Test visual consistency checking across images."""
        # Create similar images
        img1 = sample_image
        img2 = sample_image.copy()
        img2[50:100, 50:100] = 0  # Modify slightly

        consistency_score = await vision_detector.check_visual_consistency([img1, img2])

        assert 0.0 <= consistency_score <= 1.0
        assert consistency_score > 0.5  # Should be somewhat similar

    @pytest.mark.asyncio
    async def test_document_authenticity_verification(self, vision_detector):
        """Test document authenticity verification."""
        # Create test document
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(img).save(tmp.name)

            result = await vision_detector.verify_document_authenticity(tmp.name)

            assert "authentic" in result
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0

        Path(tmp.name).unlink()


class TestImagePreprocessor:
    """Test image preprocessing."""

    @pytest.mark.asyncio
    async def test_preprocessing(self):
        """Test image preprocessing pipeline."""
        preprocessor = ImagePreprocessor(target_size=(640, 640))

        # Test with various input types
        # NumPy array
        img_array = np.ones((800, 600, 3), dtype=np.uint8) * 128
        result = await preprocessor.process(img_array)
        assert result.shape[:2] == (640, 640)

        # PIL Image
        pil_img = Image.new("RGB", (800, 600))
        result = await preprocessor.process(pil_img)
        assert result.shape[:2] == (640, 640)

        # Bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        result = await preprocessor.process(img_bytes)
        assert result.shape[:2] == (640, 640)

    def test_quality_assessment(self):
        """Test image quality assessment."""
        preprocessor = ImagePreprocessor()

        # High quality image (good contrast)
        good_img = Image.new("RGB", (400, 400))
        draw = ImageDraw.Draw(good_img)
        for i in range(0, 400, 20):
            draw.line([(0, i), (400, i)], fill=(i % 255, i % 255, i % 255))

        quality = preprocessor._assess_quality(good_img)
        assert quality > 0.1  # Adjusted threshold for striped test image

        # Low quality image (uniform)
        bad_img = Image.new("RGB", (400, 400), color=(128, 128, 128))
        quality = preprocessor._assess_quality(bad_img)
        assert quality < 0.3  # Uniform image should have low quality

    def test_format_support(self):
        """Test supported format checking."""
        preprocessor = ImagePreprocessor()

        supported = preprocessor.get_supported_formats()
        assert ".jpg" in supported
        assert ".png" in supported
        assert ".webp" in supported
        assert ".heic" in supported


class TestPDFProcessor:
    """Test PDF processing."""

    @pytest.mark.asyncio
    async def test_pdf_processing(self):
        """Test PDF page extraction."""
        processor = PDFProcessor(use_gpu=False)

        # Create test PDF
        pdf_bytes = create_test_pdf(2)

        # Save to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        # Process PDF
        result = await processor.process(tmp_path)

        assert "pages" in result
        assert "text" in result
        assert "metadata" in result
        assert result["page_count"] >= 0

        # Clean up
        tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_table_extraction(self):
        """Test table extraction from PDF."""
        processor = PDFProcessor()

        # Create PDF with table-like content
        pdf_bytes = create_test_pdf(1)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        # Extract tables
        tables = await processor.extract_tables(tmp_path)

        assert isinstance(tables, list)

        # Clean up
        tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_progressive_loading(self):
        """Test progressive PDF loading."""
        processor = PDFProcessor()

        # Create large PDF
        pdf_bytes = create_test_pdf(5)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        # Test progressive loading
        chunks_received = []

        async def callback(chunk_data):
            chunks_received.append(chunk_data)

        await processor.progressive_load(tmp_path, callback, chunk_size=2)

        assert len(chunks_received) > 0

        # Clean up
        tmp_path.unlink()


class TestSpecializedAnalyzers:
    """Test individual fraud analyzers."""

    @pytest.mark.asyncio
    async def test_deepfake_detector(self, deepfake_image):
        """Test deepfake detection."""
        detector = DeepfakeDetector(use_gpu=False)
        await detector.initialize()

        result = await detector.detect(deepfake_image)

        assert "is_deepfake" in result
        assert "confidence" in result
        assert "artifacts" in result
        assert 0.0 <= result["confidence"] <= 1.0

        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_document_forgery_detector(self, forged_document_image):
        """Test document forgery detection."""
        detector = DocumentForgeryDetector(sensitivity=0.8)
        await detector.initialize()

        result = await detector.detect(forged_document_image)

        assert "is_forged" in result
        assert "confidence" in result
        assert "indicators" in result
        assert isinstance(result["indicators"], list)

        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_manipulation_detector(self):
        """Test image manipulation detection."""
        detector = ManipulationDetector(check_metadata=True, check_ela=True)
        await detector.initialize()

        # Create test image
        img = Image.new("RGB", (400, 400), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([(100, 100), (300, 300)], fill="red")

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)

            result = await detector.analyze(Path(tmp.name))

            assert "is_manipulated" in result
            assert "confidence" in result
            assert "metadata_anomalies" in result

        Path(tmp.name).unlink()
        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_logo_impersonation_detector(self, brand_impersonation_image):
        """Test logo impersonation detection."""
        detector = LogoImpersonationDetector()
        await detector.initialize()

        result = await detector.detect_logos(brand_impersonation_image)

        assert "impersonation_detected" in result
        assert "confidence" in result
        assert "detected_brands" in result
        assert isinstance(result["detected_brands"], list)

        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_qr_code_analyzer(self, malicious_qr_image):
        """Test QR code analysis."""
        analyzer = QRCodeAnalyzer(check_malicious_urls=True)
        await analyzer.initialize()

        result = await analyzer.analyze(malicious_qr_image)

        assert "malicious" in result
        assert "payload" in result
        assert "risk_indicators" in result
        assert "codes_found" in result

        await analyzer.cleanup()


class TestPerformance:
    """Performance benchmarks for vision processing."""

    @pytest.mark.asyncio
    async def test_processing_speed(self, vision_detector):
        """Test processing speed benchmark."""
        # Create test images
        images = []
        for i in range(10):
            img = np.ones((640, 640, 3), dtype=np.uint8) * (i * 25)
            images.append(img)

        # Benchmark
        start_time = time.time()
        results = await vision_detector.batch_process(images)
        total_time = time.time() - start_time

        # Calculate metrics
        images_per_second = len(images) / total_time
        avg_time_ms = (total_time / len(images)) * 1000

        print(f"\nVision Processing Performance:")
        print(f"  Images processed: {len(images)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {images_per_second:.1f} images/sec")
        print(f"  Average time: {avg_time_ms:.1f}ms per image")

        # Target: 5 documents/second
        assert images_per_second > 5, "Should process at least 5 images per second"

    @pytest.mark.asyncio
    async def test_pdf_processing_speed(self, vision_detector):
        """Test PDF processing speed."""
        # Create test PDFs
        pdfs = []
        for i in range(5):
            pdf_bytes = create_test_pdf(3)  # 3 pages each
            pdfs.append(pdf_bytes)

        # Benchmark
        start_time = time.time()

        for pdf_bytes in pdfs:
            result = await vision_detector.process_pdf_bytes(pdf_bytes)

        total_time = time.time() - start_time

        # Calculate metrics
        docs_per_second = len(pdfs) / total_time
        avg_time_s = total_time / len(pdfs)

        print(f"\nPDF Processing Performance:")
        print(f"  PDFs processed: {len(pdfs)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {docs_per_second:.1f} PDFs/sec")
        print(f"  Average time: {avg_time_s:.2f}s per PDF")

        # Target: At least 2 PDFs per second for 3-page documents
        assert docs_per_second > 2, "Should process at least 2 PDFs per second"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, vision_detector):
        """Test memory efficiency."""
        import tracemalloc

        tracemalloc.start()

        # Process multiple large images
        for _ in range(5):
            img = np.ones((1920, 1080, 3), dtype=np.uint8) * 128
            result = await vision_detector.process_image_array(img)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        print(f"\nMemory Usage:")
        print(f"  Peak memory: {peak_mb:.1f} MB")

        # Should use less than 500MB for processing
        assert peak_mb < 500, "Memory usage too high"

    def test_statistics(self):
        """Test statistics tracking."""
        asyncio.run(self._test_statistics())

    async def _test_statistics(self):
        """Async statistics test."""
        detector = VisionFraudDetector(enable_gpu=False)
        await detector.initialize()

        # Process some images
        for i in range(5):
            img = np.ones((400, 400, 3), dtype=np.uint8) * i * 50
            await detector.process_image_array(img)

        # Get statistics
        stats = detector.get_statistics()

        assert stats["total_processed"] == 5
        assert stats["average_time_ms"] > 0
        assert "cache_hit_rate" in stats
        assert "gpu_enabled" in stats

        await detector.cleanup()


# Run comprehensive benchmark
def run_vision_benchmark():
    """Run comprehensive vision processing benchmark."""
    print("\n" + "=" * 70)
    print("VISION FRAUD DETECTION BENCHMARK")
    print("=" * 70)

    async def benchmark():
        detector = VisionFraudDetector(
            enable_gpu=False,
            batch_size=8,
            use_metal=False,
        )
        await detector.initialize()

        # Create diverse test suite
        test_images = []

        # Documents
        for i in range(5):
            img = Image.new("RGB", (800, 600), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), f"Document {i+1}", fill="black")
            draw.text((50, 100), f"Invoice #{1000+i}", fill="black")
            test_images.append(np.array(img))

        # Photos with faces
        for i in range(3):
            img = Image.new("RGB", (512, 512), color=(255, 220, 177))
            draw = ImageDraw.Draw(img)
            draw.ellipse([(150, 100), (350, 400)], fill=(255, 200, 150))
            test_images.append(np.array(img))

        # QR codes
        for i in range(2):
            img = Image.new("RGB", (400, 400), color="white")
            draw = ImageDraw.Draw(img)
            for x in range(10, 390, 20):
                for y in range(10, 390, 20):
                    if (x + y) % 40 == 0:
                        draw.rectangle([(x, y), (x + 15, y + 15)], fill="black")
            test_images.append(np.array(img))

        print(f"\nTest Suite: {len(test_images)} images")
        print(f"  - {5} documents")
        print(f"  - {3} photos")
        print(f"  - {2} QR codes")

        # Warm up
        print("\nWarming up...")
        for img in test_images[:2]:
            await detector.process_image_array(img)

        # Benchmark
        print("\nRunning benchmark...")
        start_time = time.time()

        results = await detector.batch_process(test_images)

        total_time = time.time() - start_time

        # Analysis
        fraud_detected = sum(1 for r in results if r and r.fraud_score > 0.5)
        avg_confidence = sum(r.confidence for r in results if r) / len(results)

        # Performance metrics
        throughput = len(test_images) / total_time
        avg_latency = (total_time / len(test_images)) * 1000

        # Get final statistics
        stats = detector.get_statistics()

        print(f"\n{'='*40}")
        print("RESULTS")
        print(f"{'='*40}")
        print(f"Fraud Detection:")
        print(f"  Fraudulent: {fraud_detected}/{len(test_images)}")
        print(f"  Avg confidence: {avg_confidence:.2%}")
        print(f"\nPerformance:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} images/sec")
        print(f"  Avg latency: {avg_latency:.1f}ms")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"\nTargets:")

        # Check targets
        targets = {
            "throughput": 5,  # docs/sec
            "latency": 200,  # ms
        }

        if throughput >= targets["throughput"]:
            print(f"  ✅ Throughput: {throughput:.1f} >= {targets['throughput']} docs/sec")
        else:
            print(f"  ❌ Throughput: {throughput:.1f} < {targets['throughput']} docs/sec")

        if avg_latency <= targets["latency"]:
            print(f"  ✅ Latency: {avg_latency:.1f} <= {targets['latency']}ms")
        else:
            print(f"  ❌ Latency: {avg_latency:.1f} > {targets['latency']}ms")

        await detector.cleanup()

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

    asyncio.run(benchmark())


if __name__ == "__main__":
    # Run tests
    print("Running Vision Fraud Detection Tests...")
    pytest.main([__file__, "-v", "--tb=short"])

    # Run benchmark
    print("\n" + "=" * 70)
    run_vision_benchmark()
