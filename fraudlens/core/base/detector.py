"""
Base fraud detector abstract class.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class FraudType(Enum):
    """Enumeration of fraud types."""
    
    IDENTITY_THEFT = "identity_theft"
    PAYMENT_FRAUD = "payment_fraud"
    DOCUMENT_FORGERY = "document_forgery"
    SYNTHETIC_CONTENT = "synthetic_content"
    ACCOUNT_TAKEOVER = "account_takeover"
    MONEY_LAUNDERING = "money_laundering"
    PHISHING = "phishing"
    SCAM = "scam"
    DEEPFAKE = "deepfake"
    IMAGE_MANIPULATION = "image_manipulation"
    BRAND_IMPERSONATION = "brand_impersonation"
    MALICIOUS_QR = "malicious_qr"
    SOCIAL_ENGINEERING = "social_engineering"
    UNKNOWN = "unknown"


class Modality(Enum):
    """Supported input modalities."""
    
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class DetectionResult:
    """Result from fraud detection analysis."""
    
    fraud_score: float  # 0.0 to 1.0
    fraud_types: List[FraudType]
    confidence: float  # 0.0 to 1.0
    explanation: str
    evidence: Dict[str, Any]
    timestamp: datetime
    detector_id: str
    modality: Modality
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None
    
    def is_fraud(self, threshold: float = 0.5) -> bool:
        """Check if result indicates fraud based on threshold."""
        return self.fraud_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "fraud_score": self.fraud_score,
            "fraud_types": [ft.value for ft in self.fraud_types],
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "detector_id": self.detector_id,
            "modality": self.modality.value,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata or {}
        }


class FraudDetector(ABC):
    """Abstract base class for all fraud detectors."""
    
    def __init__(
        self,
        detector_id: str,
        modality: Modality,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[Path] = None,
        device: str = "mps",  # Default to Apple Silicon GPU
    ):
        """
        Initialize fraud detector.
        
        Args:
            detector_id: Unique identifier for this detector
            modality: Input modality this detector supports
            config: Configuration dictionary
            model_path: Path to model weights/files
            device: Compute device ('mps', 'cpu', 'cuda')
        """
        self.detector_id = detector_id
        self.modality = modality
        self.config = config or {}
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize detector and load models."""
        pass
    
    @abstractmethod
    async def detect(
        self,
        input_data: Union[str, bytes, np.ndarray, Path],
        **kwargs
    ) -> DetectionResult:
        """
        Perform fraud detection on input data.
        
        Args:
            input_data: Input to analyze (format depends on modality)
            **kwargs: Additional detector-specific parameters
            
        Returns:
            DetectionResult containing fraud analysis
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and unload models."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.
        
        Returns:
            Memory usage in bytes
        """
        pass
    
    @abstractmethod
    def validate_input(
        self,
        input_data: Union[str, bytes, np.ndarray, Path]
    ) -> bool:
        """
        Validate input data format and content.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get detector information and capabilities."""
        return {
            "detector_id": self.detector_id,
            "modality": self.modality.value,
            "initialized": self._initialized,
            "device": self.device,
            "model_path": str(self.model_path) if self.model_path else None,
            "config": self.config,
        }
    
    def __repr__(self) -> str:
        """String representation of detector."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self.detector_id}', "
            f"modality={self.modality.value}, "
            f"initialized={self._initialized})"
        )