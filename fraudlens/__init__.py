"""
FraudLens: Multi-modal fraud detection system optimized for Apple Silicon.

A production-ready, extensible fraud detection framework supporting text, image,
PDF, video, and audio analysis with plugin architecture and resource management.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

__version__ = "0.1.0"
__author__ = "Yobie Benjamin"
__license__ = "Apache-2.0"

from fraudlens.core.base.detector import FraudDetector
from fraudlens.core.base.processor import ModalityProcessor
from fraudlens.core.base.scorer import RiskScorer
from fraudlens.core.registry.model_registry import ModelRegistry
from fraudlens.core.resource_manager.manager import ResourceManager

__all__ = [
    "FraudDetector",
    "ModalityProcessor",
    "RiskScorer",
    "ModelRegistry",
    "ResourceManager",
]