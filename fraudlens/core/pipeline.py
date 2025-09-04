"""
Fraud Detection Pipeline orchestrating all components.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from fraudlens.core.base.detector import DetectionResult, Modality
from fraudlens.core.config import Config
from fraudlens.core.registry.model_registry import ModelRegistry
from fraudlens.core.resource_manager.manager import ResourceManager
from fraudlens.integrations import (
    DocumentValidator,
    PhishingDatabaseConnector,
    ThreatIntelligenceManager,
)
from fraudlens.plugins.manager import PluginManager
from fraudlens.processors.text.detector import TextFraudDetector
from fraudlens.processors.vision.detector import VisionFraudDetector


class FraudDetectionPipeline:
    """
    Main pipeline orchestrating fraud detection across modalities.

    This is the primary entry point for fraud detection, coordinating:
    - Multiple modality processors
    - Resource management
    - Plugin execution
    - Model registry
    - Caching and optimization
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize fraud detection pipeline.

        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self._initialized = False

        # Core components
        self.resource_manager = ResourceManager(
            max_memory_gb=self.config.get("resource_limits.max_memory_gb", 100),
        )

        self.model_registry = ModelRegistry()
        self.plugin_manager = PluginManager()

        # External integrations
        config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        self.threat_intel = ThreatIntelligenceManager(config=config_dict)
        self.document_validator = DocumentValidator()
        self.phishing_db = PhishingDatabaseConnector(config=config_dict)

        # Processors for different modalities
        self.processors = {}
        self._init_processors()

        # Statistics
        self._total_processed = 0
        self._total_time_ms = 0
        self._errors = 0

    def _init_processors(self):
        """Initialize modality processors based on config."""
        # Text processor (70% of workload)
        if self.config.get("processors.text.enabled", True):
            self.processors["text"] = TextFraudDetector(
                cache_size=self.config.get("cache.max_size", 1000),
                enable_gpu=self.config.get("resource_limits.enable_gpu", True),
                batch_size=self.config.get("processors.text.batch_size", 32),
            )

        # Vision processor (images and PDFs)
        if self.config.get("processors.vision.enabled", True):
            self.processors["image"] = VisionFraudDetector(
                enable_gpu=self.config.get("resource_limits.enable_gpu", True),
                batch_size=self.config.get("processors.vision.batch_size", 8),
                cache_size=self.config.get("cache.max_size", 100),
                use_metal=self.config.get("processors.vision.use_metal", True),
            )
            # PDF uses same processor
            self.processors["pdf"] = self.processors["image"]

        # TODO: Add video, audio processors when implemented

    async def initialize(self):
        """Initialize pipeline and all components."""
        if self._initialized:
            return

        logger.info("Initializing FraudDetection Pipeline...")
        start_time = time.time()

        # Start resource monitoring
        await self.resource_manager.start_monitoring()

        # Initialize processors
        init_tasks = []
        for name, processor in self.processors.items():
            if hasattr(processor, "initialize"):
                init_tasks.append(processor.initialize())

        if init_tasks:
            await asyncio.gather(*init_tasks)

        # Load plugins
        await self.plugin_manager.load_plugins()

        # Initialize external integrations
        await self.threat_intel.initialize()
        await self.phishing_db.check_url("https://example.com")  # Prime cache

        self._initialized = True
        init_time = (time.time() - start_time) * 1000
        logger.info(f"Pipeline initialized in {init_time:.0f}ms")

    async def process(
        self, input_data: Union[str, bytes, Path], modality: Optional[str] = None, **kwargs
    ) -> Optional[DetectionResult]:
        """
        Process input through fraud detection pipeline.

        Args:
            input_data: Input data to process
            modality: Modality type (text, image, video, audio)
            **kwargs: Additional arguments

        Returns:
            Detection result or None if processing fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Auto-detect modality if not specified
            if modality is None:
                modality = self._detect_modality(input_data)

            # Check resource limits
            if not await self.resource_manager.request_memory(100):  # 100MB buffer
                logger.warning("Insufficient memory for processing")
                return None

            # Get appropriate processor
            processor = self.processors.get(modality)
            if processor is None:
                logger.warning(f"No processor available for modality: {modality}")
                return self._create_empty_result(modality)

            # Process with resource tracking
            start_time = time.time()

            # Execute processing
            result = await processor.detect(input_data, **kwargs)

            # Track statistics
            processing_time = (time.time() - start_time) * 1000
            self._total_processed += 1
            self._total_time_ms += processing_time

            # Execute plugins if configured
            if self.plugin_manager.has_plugins():
                plugin_results = await self.plugin_manager.execute_plugins(
                    input_data, result, modality=modality
                )
                # Merge plugin results
                if plugin_results:
                    result = self._merge_plugin_results(result, plugin_results)

            return result

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            self._errors += 1
            return self._create_error_result(str(e), modality)

    async def batch_process(
        self, inputs: List[Union[str, bytes, Path]], modality: Optional[str] = None, **kwargs
    ) -> List[Optional[DetectionResult]]:
        """
        Process batch of inputs.

        Args:
            inputs: List of inputs to process
            modality: Modality type
            **kwargs: Additional arguments

        Returns:
            List of detection results
        """
        if not self._initialized:
            await self.initialize()

        # Auto-detect modality from first input
        if modality is None and inputs:
            modality = self._detect_modality(inputs[0])

        # Get processor
        processor = self.processors.get(modality)
        if processor is None:
            return [self._create_empty_result(modality) for _ in inputs]

        # Use batch processing if available
        if hasattr(processor, "batch_process"):
            return await processor.batch_process(inputs, **kwargs)
        else:
            # Fall back to sequential processing
            tasks = [self.process(inp, modality, **kwargs) for inp in inputs]
            return await asyncio.gather(*tasks, return_exceptions=True)

    def _detect_modality(self, input_data: Any) -> str:
        """Auto-detect modality from input."""
        # Check for numpy arrays (images)
        try:
            import numpy as np

            if isinstance(input_data, np.ndarray):
                # Assume image if it's a 2D or 3D array
                if len(input_data.shape) in [2, 3]:
                    return "image"
                return "text"  # Default for other arrays
        except ImportError:
            pass

        if isinstance(input_data, str):
            # Check if it's a file path
            try:
                path = Path(input_data)
                if path.exists():
                    return self._detect_modality(path)
            except:
                pass
            return "text"
        elif isinstance(input_data, Path):
            suffix = input_data.suffix.lower()
            if suffix in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic"]:
                return "image"
            elif suffix == ".pdf":
                return "pdf"
            elif suffix in [".mp4", ".avi", ".mov", ".wmv"]:
                return "video"
            elif suffix in [".mp3", ".wav", ".flac", ".aac"]:
                return "audio"
            else:
                return "text"  # Default to text
        elif isinstance(input_data, bytes):
            # Check magic bytes
            if input_data[:4] == b"%PDF":
                return "pdf"
            elif input_data[:2] in [b"\xff\xd8", b"\x89P"]:  # JPEG or PNG
                return "image"
            else:
                return "text"
        else:
            return "text"  # Default

    def _create_empty_result(self, modality: str) -> DetectionResult:
        """Create empty detection result."""
        return DetectionResult(
            fraud_score=0.0,
            fraud_types=[],
            confidence=0.0,
            explanation="No processor available for this modality",
            evidence={},
            timestamp=datetime.now(),
            detector_id="pipeline",
            modality=Modality.TEXT if modality == "text" else Modality.IMAGE,
            processing_time_ms=0,
            metadata={"error": "no_processor"},
        )

    def _create_error_result(self, error: str, modality: str) -> DetectionResult:
        """Create error detection result."""
        return DetectionResult(
            fraud_score=0.0,
            fraud_types=[],
            confidence=0.0,
            explanation=f"Processing error: {error}",
            evidence={},
            timestamp=datetime.now(),
            detector_id="pipeline",
            modality=Modality.TEXT if modality == "text" else Modality.IMAGE,
            processing_time_ms=0,
            metadata={"error": error},
        )

    def _merge_plugin_results(
        self, result: DetectionResult, plugin_results: List[Dict]
    ) -> DetectionResult:
        """Merge plugin results with main result."""
        # Simple averaging for now
        if plugin_results:
            plugin_scores = [r.get("fraud_score", 0) for r in plugin_results]
            avg_plugin_score = sum(plugin_scores) / len(plugin_scores)

            # Weight: 70% original, 30% plugins
            result.fraud_score = (result.fraud_score * 0.7) + (avg_plugin_score * 0.3)

            # Add plugin evidence
            result.evidence["plugin_results"] = plugin_results

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_time = self._total_time_ms / self._total_processed if self._total_processed > 0 else 0

        return {
            "total_processed": self._total_processed,
            "average_time_ms": avg_time,
            "total_time_ms": self._total_time_ms,
            "errors": self._errors,
            "error_rate": self._errors / self._total_processed if self._total_processed > 0 else 0,
            "resource_stats": self.resource_manager.get_statistics(),
            "processors": list(self.processors.keys()),
        }

    async def validate_document(
        self, document_data: str, document_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Validate identity document using external databases.

        Args:
            document_data: Document number or MRZ data
            document_type: Type of document (ssn, passport, driver_license, etc.)

        Returns:
            Validation result with fraud indicators
        """
        # Use document validator
        validation = self.document_validator.validate_document(document_data, document_type)

        # Enhance with fraud scoring
        fraud_score = 0.0
        if not validation.get("valid", False):
            fraud_score = 0.8  # Invalid documents are highly suspicious

        return {**validation, "fraud_score": fraud_score, "timestamp": datetime.now().isoformat()}

    async def check_url_threat(self, url: str) -> Dict[str, Any]:
        """
        Check URL against threat intelligence databases.

        Args:
            url: URL to check

        Returns:
            Threat assessment with recommendations
        """
        # Check phishing databases
        phishing_result = await self.phishing_db.check_url(url)

        # Check threat intelligence
        threat_result = await self.threat_intel.check_url(url)

        # Combine results
        combined_score = max(
            phishing_result.get("confidence", 0.0), threat_result.get("threat_score", 0.0)
        )

        return {
            "url": url,
            "is_malicious": combined_score > 0.5,
            "threat_score": combined_score,
            "phishing_analysis": phishing_result,
            "threat_intelligence": threat_result,
            "recommendations": phishing_result.get("recommendations", []),
        }

    async def check_email_threat(self, email: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Check email for fraud indicators.

        Args:
            email: Email address
            content: Email content (optional)

        Returns:
            Fraud assessment
        """
        # Check email address
        email_threat = await self.threat_intel.check_email(email)

        fraud_score = email_threat.get("threat_score", 0.0)
        threats = email_threat.get("threats", [])

        # If content provided, check for phishing
        if content:
            text_result = await self.process(content, modality="text")
            if text_result:
                fraud_score = max(fraud_score, text_result.fraud_score / 100)
                if text_result.fraud_types:
                    threats.extend([f"Content: {ft}" for ft in text_result.fraud_types])

        return {
            "email": email,
            "fraud_score": fraud_score,
            "threats": threats,
            "is_suspicious": fraud_score > 0.5,
        }

    async def enhance_detection_with_intel(self, result: DetectionResult) -> DetectionResult:
        """
        Enhance detection result with threat intelligence.

        Args:
            result: Original detection result

        Returns:
            Enhanced detection result
        """
        # Extract URLs from result if present
        if hasattr(result, "metadata") and "urls" in result.metadata:
            for url in result.metadata["urls"]:
                threat_result = await self.check_url_threat(url)
                if threat_result["is_malicious"]:
                    result.fraud_score = min(100, result.fraud_score + 20)
                    result.fraud_types.append("malicious_url")

        return result

    async def cleanup(self):
        """Clean up pipeline resources."""
        logger.info("Cleaning up FraudDetection Pipeline...")

        # Clean up processors
        cleanup_tasks = []
        for processor in self.processors.values():
            if hasattr(processor, "cleanup"):
                cleanup_tasks.append(processor.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Clean up integrations
        await self.threat_intel.cleanup()
        await self.phishing_db.cleanup()

        # Clean up other components
        await self.resource_manager.stop_monitoring()
        await self.plugin_manager.cleanup()

        self._initialized = False
        logger.info("Pipeline cleanup complete")
