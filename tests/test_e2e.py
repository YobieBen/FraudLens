"""
End-to-end tests for FraudLens fraud detection system.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraudlens.core.base.detector import (
    DetectionResult,
    FraudDetector,
    FraudType,
    Modality,
)
from fraudlens.core.base.processor import ModalityProcessor, ProcessedData
from fraudlens.core.base.scorer import RiskAssessment, RiskLevel, RiskScorer
from fraudlens.core.registry.model_registry import ModelFormat, ModelRegistry, QuantizationType
from fraudlens.core.resource_manager.manager import ResourceManager
from fraudlens.pipelines.async_pipeline import AsyncPipeline
from fraudlens.plugins.base import FraudLensPlugin, PluginMetadata
from fraudlens.plugins.loader import PluginLoader
from fraudlens.utils.config import ConfigManager


class TestFraudDetector(FraudDetector):
    """Test implementation of FraudDetector."""

    def __init__(self):
        super().__init__(
            detector_id="test_detector", modality=Modality.TEXT, config={"threshold": 0.5}
        )

    async def initialize(self) -> None:
        self._initialized = True

    async def detect(self, input_data: str, **kwargs) -> DetectionResult:
        await asyncio.sleep(0.1)  # Simulate processing

        # Handle both string and array inputs
        if isinstance(input_data, np.ndarray):
            input_data = str(input_data)
        elif not isinstance(input_data, str):
            input_data = str(input_data)

        # Simple keyword-based detection for testing
        fraud_keywords = ["fraud", "scam", "urgent", "click here"]
        score = sum(1 for keyword in fraud_keywords if keyword in input_data.lower()) * 0.25

        return DetectionResult(
            fraud_score=min(score, 1.0),
            fraud_types=[FraudType.PHISHING if score > 0 else FraudType.UNKNOWN],
            confidence=0.8,
            explanation="Test detection completed",
            evidence={"keywords_found": score > 0},
            timestamp=asyncio.get_event_loop().time(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=100,
        )

    async def cleanup(self) -> None:
        self._initialized = False

    def get_memory_usage(self) -> int:
        return 50 * 1024 * 1024  # 50MB

    def validate_input(self, input_data: str) -> bool:
        return isinstance(input_data, str) and len(input_data) > 0


class TestProcessor(ModalityProcessor):
    """Test implementation of ModalityProcessor."""

    def __init__(self):
        super().__init__(modality="text", config={})

    async def process(self, input_data: str, **kwargs) -> ProcessedData:
        await asyncio.sleep(0.05)  # Simulate processing

        # Simple tokenization for testing
        tokens = input_data.lower().split()
        data = np.array([len(token) for token in tokens])

        return ProcessedData(
            data=data,
            original_shape=(len(tokens),),
            metadata={"token_count": len(tokens)},
            preprocessing_steps=["tokenize", "vectorize"],
        )

    async def validate(self, input_data: str):
        is_valid = isinstance(input_data, str) and len(input_data) > 0
        error = None if is_valid else "Invalid input"
        return is_valid, error

    def get_output_shape(self):
        return (None,)  # Variable length

    def get_preprocessing_steps(self):
        return ["tokenize", "vectorize"]


class TestScorer(RiskScorer):
    """Test implementation of RiskScorer."""

    def __init__(self):
        super().__init__(scorer_id="test_scorer")

    async def score(self, detection_results: List[DetectionResult], **kwargs) -> RiskAssessment:
        if not detection_results:
            overall_score = 0.0
        else:
            scores = [r.fraud_score for r in detection_results]
            overall_score = self.aggregate_scores(scores)

        risk_level = RiskLevel.from_score(overall_score)

        factors = [
            {
                "detector": r.detector_id,
                "score": r.fraud_score,
                "weight": 1.0,
            }
            for r in detection_results
            if r.fraud_score > 0
        ]

        recommendations = self.generate_recommendations(
            RiskAssessment(
                overall_score=overall_score,
                risk_level=risk_level,
                confidence=0.85,
                contributing_factors=factors,
                recommendations=[],
                timestamp=asyncio.get_event_loop().time(),
                assessment_id="test_assessment",
            )
        )

        return RiskAssessment(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=self.calculate_confidence(detection_results),
            contributing_factors=factors,
            recommendations=recommendations,
            timestamp=asyncio.get_event_loop().time(),
            assessment_id="test_assessment",
        )

    def aggregate_scores(self, scores: List[float], weights: List[float] = None) -> float:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def calculate_confidence(self, detection_results: List[DetectionResult]) -> float:
        if not detection_results:
            return 0.0
        return sum(r.confidence for r in detection_results) / len(detection_results)

    def generate_recommendations(self, risk_assessment: RiskAssessment) -> List[str]:
        if risk_assessment.risk_level == RiskLevel.VERY_HIGH:
            return ["Block transaction", "Manual review required"]
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            return ["Flag for review"]
        else:
            return ["Continue monitoring"]


class TestPlugin(FraudLensPlugin):
    """Test plugin implementation."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="Test plugin for E2E testing",
            dependencies=[],
            fraudlens_version="0.1.0",
            license="Apache-2.0",
            tags=["test"],
        )

    async def initialize(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    def get_detectors(self):
        return {"test_detector": TestFraudDetector}


class TestE2EFraudLens:
    """End-to-end test suite for FraudLens."""

    @pytest.mark.asyncio
    async def test_01_module_imports(self):
        """Test that all core modules can be imported."""
        print("\n=== Testing Module Imports ===")

        # Test core imports
        from fraudlens.core.base import detector, processor, scorer
        from fraudlens.core.registry import model_registry
        from fraudlens.core.resource_manager import manager
        from fraudlens.pipelines import async_pipeline
        from fraudlens.plugins import base, loader
        from fraudlens.utils import config, logging

        assert detector.FraudDetector is not None
        assert processor.ModalityProcessor is not None
        assert scorer.RiskScorer is not None
        assert manager.ResourceManager is not None
        assert model_registry.ModelRegistry is not None
        assert async_pipeline.AsyncPipeline is not None
        assert base.FraudLensPlugin is not None
        assert loader.PluginLoader is not None
        assert config.ConfigManager is not None
        assert logging.get_logger is not None

        print("✓ All core modules imported successfully")

    @pytest.mark.asyncio
    async def test_02_fraud_detector(self):
        """Test FraudDetector base class and implementation."""
        print("\n=== Testing FraudDetector ===")

        detector = TestFraudDetector()

        # Test initialization
        assert detector.detector_id == "test_detector"
        assert detector.modality == Modality.TEXT
        assert not detector._initialized

        await detector.initialize()
        assert detector._initialized

        # Test detection
        result = await detector.detect("This is a urgent scam message! Click here now!")
        assert isinstance(result, DetectionResult)
        assert result.fraud_score > 0
        assert result.fraud_types[0] == FraudType.PHISHING
        assert result.confidence == 0.8
        assert result.detector_id == "test_detector"

        # Test validation
        assert detector.validate_input("valid text")
        assert not detector.validate_input("")

        # Test cleanup
        await detector.cleanup()
        assert not detector._initialized

        print(f"✓ FraudDetector tested - Score: {result.fraud_score:.2f}")

    @pytest.mark.asyncio
    async def test_03_modality_processor(self):
        """Test ModalityProcessor functionality."""
        print("\n=== Testing ModalityProcessor ===")

        processor = TestProcessor()

        # Test processing
        processed = await processor.process("This is a test message")
        assert isinstance(processed, ProcessedData)
        assert isinstance(processed.data, np.ndarray)
        assert processed.metadata["token_count"] == 5

        # Test validation
        is_valid, error = await processor.validate("valid input")
        assert is_valid
        assert error is None

        is_valid, error = await processor.validate("")
        assert not is_valid
        assert error == "Invalid input"

        # Test preprocessing steps
        steps = processor.get_preprocessing_steps()
        assert "tokenize" in steps
        assert "vectorize" in steps

        print(f"✓ ModalityProcessor tested - Tokens: {processed.metadata['token_count']}")

    @pytest.mark.asyncio
    async def test_04_risk_scorer(self):
        """Test RiskScorer aggregation."""
        print("\n=== Testing RiskScorer ===")

        scorer = TestScorer()
        detector = TestFraudDetector()
        await detector.initialize()

        # Generate detection results
        results = []
        for text in ["normal text", "urgent scam!", "click here for fraud"]:
            result = await detector.detect(text)
            results.append(result)

        # Test scoring
        assessment = await scorer.score(results)
        assert isinstance(assessment, RiskAssessment)
        assert 0 <= assessment.overall_score <= 1
        assert assessment.risk_level in RiskLevel
        assert len(assessment.recommendations) > 0
        assert assessment.confidence > 0

        # Test empty results
        empty_assessment = await scorer.score([])
        assert empty_assessment.overall_score == 0.0
        assert empty_assessment.risk_level == RiskLevel.VERY_LOW

        await detector.cleanup()

        print(
            f"✓ RiskScorer tested - Risk: {assessment.risk_level.value}, Score: {assessment.overall_score:.2f}"
        )

    @pytest.mark.asyncio
    async def test_05_resource_manager(self):
        """Test ResourceManager monitoring."""
        print("\n=== Testing ResourceManager ===")

        manager = ResourceManager(
            max_memory_gb=100,
            warning_threshold=0.7,
            enable_monitoring=False,  # Disable background monitoring for test
        )

        # Test snapshot
        snapshot = manager.get_snapshot()
        assert snapshot.memory_used_gb >= 0
        assert snapshot.memory_available_gb >= 0
        assert snapshot.active_models == 0

        # Test model registration
        manager.register_model("test_model", object(), estimated_memory_mb=100)
        assert "test_model" in manager._active_models
        assert manager._model_memory["test_model"] == 100 * 1024 * 1024

        # Test model unregistration
        manager.unregister_model("test_model")
        assert "test_model" not in manager._active_models

        # Test statistics
        stats = manager.get_statistics()
        assert "current" in stats
        assert "config" in stats
        assert stats["config"]["max_memory_gb"] == 100

        print(
            f"✓ ResourceManager tested - Memory: {snapshot.memory_used_gb:.2f}GB / {manager.max_memory_gb}GB"
        )

    @pytest.mark.asyncio
    async def test_06_async_pipeline(self):
        """Test AsyncPipeline processing."""
        print("\n=== Testing AsyncPipeline ===")

        pipeline = AsyncPipeline(max_workers=3, batch_size=10)

        # Register components
        detector = TestFraudDetector()
        processor = TestProcessor()
        scorer = TestScorer()

        pipeline.register_detector("test_detector", detector)
        pipeline.register_processor(Modality.TEXT, processor)
        pipeline.register_scorer("test_scorer", scorer)

        # Start pipeline
        await pipeline.start()
        assert pipeline._running

        # Process single input
        result = await pipeline.process(
            "This is a fraud scam message", modality=Modality.TEXT, wait=True
        )
        assert result.task_id is not None
        assert len(result.detection_results) > 0
        assert result.risk_assessment is not None
        assert result.is_successful()

        # Process batch
        batch_inputs = [
            "Normal message",
            "Urgent scam alert!",
            "Click here now fraud",
        ]
        batch_results = await pipeline.process(batch_inputs, modality=Modality.TEXT, wait=True)
        assert len(batch_results) == 3
        assert all(r.is_successful() for r in batch_results)

        # Test statistics
        stats = pipeline.get_statistics()
        assert stats["total_processed"] >= 4
        assert stats["active_tasks"] == 0
        assert "detectors" in stats

        # Stop pipeline
        await pipeline.stop()
        assert not pipeline._running

        print(f"✓ AsyncPipeline tested - Processed: {stats['total_processed']} tasks")

    @pytest.mark.asyncio
    async def test_07_config_manager(self):
        """Test ConfigManager functionality."""
        print("\n=== Testing ConfigManager ===")

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
system:
  max_memory_gb: 50
  device: cpu
detection:
  threshold: 0.6
"""
            )
            config_path = f.name

        try:
            manager = ConfigManager(config_path=config_path)
            config = manager.load()

            # Test get
            assert manager.get("system.max_memory_gb") == 50
            assert manager.get("system.device") == "cpu"
            assert manager.get("detection.threshold") == 0.6
            assert manager.get("nonexistent", "default") == "default"

            # Test set
            manager.set("new_key", "new_value")
            assert manager.get("new_key") == "new_value"

            # Test nested set
            manager.set("nested.key.value", 42)
            assert manager.get("nested.key.value") == 42

            print("✓ ConfigManager tested - Config loaded and modified")

        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_08_plugin_system(self):
        """Test Plugin system."""
        print("\n=== Testing Plugin System ===")

        # Create plugin instance
        plugin = TestPlugin()

        # Test metadata
        metadata = plugin.get_metadata()
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"

        # Test initialization
        await plugin.initialize({})
        assert plugin._initialized

        # Test detector provision
        detectors = plugin.get_detectors()
        assert "test_detector" in detectors
        assert detectors["test_detector"] == TestFraudDetector

        # Test cleanup
        await plugin.cleanup()
        assert not plugin._initialized

        print(f"✓ Plugin system tested - Plugin: {metadata.name} v{metadata.version}")

    @pytest.mark.asyncio
    async def test_09_model_registry(self):
        """Test ModelRegistry functionality."""
        print("\n=== Testing ModelRegistry ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=Path(tmpdir))

            # Create a dummy model file
            model_path = Path(tmpdir) / "test_model.onnx"
            model_path.write_bytes(b"dummy model content")

            # Register model
            model_info = registry.register_model(
                name="test_model",
                path=model_path,
                format=ModelFormat.ONNX,
                modality="text",
                version="1.0.0",
                quantization=QuantizationType.INT8,
                tags=["test", "demo"],
            )

            assert model_info.name == "test_model"
            assert model_info.version == "1.0.0"
            assert model_info.format == ModelFormat.ONNX
            assert model_info.quantization == QuantizationType.INT8
            assert "test" in model_info.tags

            # Get model
            retrieved = registry.get_model(model_id=model_info.model_id)
            assert retrieved is not None
            assert retrieved.name == "test_model"

            # List models
            models = registry.list_models(modality="text")
            assert len(models) == 1
            assert models[0].name == "test_model"

            # Test statistics
            stats = registry.get_statistics()
            assert stats["total_models"] == 1
            assert "text" in stats["modalities"]

            print(f"✓ ModelRegistry tested - Models: {stats['total_models']}")

    @pytest.mark.asyncio
    async def test_10_full_e2e_flow(self):
        """Test complete end-to-end flow."""
        print("\n=== Testing Full E2E Flow ===")

        # Initialize all components
        resource_manager = ResourceManager(max_memory_gb=100, enable_monitoring=False)
        pipeline = AsyncPipeline(max_workers=5)
        detector = TestFraudDetector()
        scorer = TestScorer()

        # Setup pipeline
        pipeline.register_detector("detector", detector)
        pipeline.register_scorer("scorer", scorer)
        await pipeline.start()

        # Test data
        test_samples = [
            ("Normal transaction", False),
            ("URGENT: Click here to claim your prize! This is a scam!", True),
            ("Your order has been shipped", False),
            ("Fraud alert: Suspicious activity detected", True),
        ]

        results = []
        for text, is_fraud in test_samples:
            result = await pipeline.process(text, modality=Modality.TEXT)
            results.append((text[:30], is_fraud, result))

        # Verify results
        correct_predictions = 0
        for text, expected_fraud, result in results:
            is_fraud_detected = result.risk_assessment.overall_score > 0.3
            if is_fraud_detected == expected_fraud:
                correct_predictions += 1

            status = "✓" if is_fraud_detected == expected_fraud else "✗"
            print(
                f"  {status} '{text}...' - Score: {result.risk_assessment.overall_score:.2f}, "
                f"Expected fraud: {expected_fraud}, Detected: {is_fraud_detected}"
            )

        accuracy = correct_predictions / len(results)

        # Cleanup
        await pipeline.stop()

        print(f"\n✓ Full E2E flow tested - Accuracy: {accuracy:.1%}")
        assert accuracy >= 0.75  # At least 75% accuracy


def run_all_tests():
    """Run all E2E tests."""
    print("\n" + "=" * 60)
    print("FRAUDLENS END-TO-END TEST SUITE")
    print("=" * 60)

    # Run tests
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_all_tests()
